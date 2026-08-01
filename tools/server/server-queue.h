#pragma once

#include "server-task.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>
#include <unordered_set>

// struct for managing server tasks
// in most cases, use server_response_reader to post new tasks and retrieve results
struct server_queue {
private:
    int id = 0;
    bool running  = false;
    bool sleeping = false;
    bool req_stop_sleeping = false;
    // pre-task ctx window tracking (--rpc-reload segfault fix): HTTP threads
    // read the model (tokenize, chat templates) between create_response() and
    // post_tasks(); teardown must not run inside that window
    bool ctx_hold = false;      // a teardown holds the model: block new guards
    int  n_ctx_guards = 0;      // HTTP threads currently inside the window
    int64_t time_last_task = 0;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(server_task &&)> callback_new_task;
    std::function<void(void)>           callback_update_slots;
    std::function<void(bool)>           callback_sleeping_state;

public:
    // Add a new task to the end of the queue
    int post(server_task && task, bool front = false);

    // multi-task version of post()
    int post(std::vector<server_task> && tasks, bool front = false);

    // Add a new task, but defer until one slot is available
    void defer(server_task && task);

    // Get the next id for creating a new task
    int get_new_id();

    // Call when the state of one slot is changed, it will move one task from deferred to main queue
    // prioritize tasks that use the specified slot (otherwise, pop the first deferred task)
    void pop_deferred_task(int id_slot);

    // if sleeping, request exiting sleep state and wait until it is done
    // returns immediately if not sleeping
    void wait_until_no_sleep();

    // pre-task ctx guard: enter blocks while a teardown hold is active (and
    // wakes/waits out the sleeping state, subsuming wait_until_no_sleep);
    // exit releases. Use via server_response_reader::ctx_guard_acquire/release.
    // hold_timeout_ms bounds only the ctx_hold wait (< 0 = wait forever):
    // a wedged --rpc-reload retry loop holds the ctx indefinitely, and parking
    // HTTP threads on it untimed exhausts the pool (code review 2026-08-01 #1).
    // Returns false if the hold outlasted the timeout; the guard is NOT held.
    bool ctx_guard_enter(int hold_timeout_ms = -1);
    void ctx_guard_exit();
    // teardown side (queue thread): block new guards, then wait for in-flight
    // ones to drain (bounded); end releases the hold
    void ctx_hold_begin(int timeout_ms);
    void ctx_hold_end();

    bool is_sleeping() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return sleeping;
    }

    // end the start_loop routine
    void terminate();

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     *
     * Sleeping procedure (disabled if idle_sleep_ms < 0):
     * - If there is no task after idle_sleep_ms, enter sleeping state
     * - Call callback_sleeping_state(true)
     * - Wait until req_stop_sleeping is set to true
     * - Call callback_sleeping_state(false)
     * - Exit sleeping state
     */
    void start_loop(int64_t idle_sleep_ms = -1);

    // for metrics
    size_t queue_tasks_deferred_size() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return queue_tasks_deferred.size();
    }

    //
    // Functions below are not thread-safe, must only be used before start_loop() is called
    //

    // Register function to process a new task
    void on_new_task(std::function<void(server_task &&)> callback) {
        callback_new_task = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Register callback for sleeping state change; multiple callbacks are allowed
    // note: when entering sleeping state, the callback is called AFTER sleeping is set to true
    //       when leaving sleeping state, the callback is called BEFORE sleeping is set to false
    void on_sleeping_state(std::function<void(bool)> callback) {
        if (callback_sleeping_state) {
            auto prev_callback = std::move(callback_sleeping_state);
            callback_sleeping_state = [prev_callback, callback](bool sleeping) {
                prev_callback(sleeping);
                callback(sleeping);
            };
        } else {
            callback_sleeping_state = std::move(callback);
        }
    }

private:
    void cleanup_pending_task(int id_target);
};

// struct for managing server responses
// in most cases, use server_response_reader to retrieve results
struct server_response {
private:
    bool running = true;

    // for keeping track of all tasks waiting for the result
    std::unordered_set<int> waiting_task_ids;

    // the main result queue (using ptr for polymorphism)
    std::vector<server_task_result_ptr> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;

public:
    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task);

    void add_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task);

    // remove multiple tasks from waiting list
    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // This function blocks the thread until there is a response for one of the id_tasks
    server_task_result_ptr recv(const std::unordered_set<int> & id_tasks);

    // same as recv(), but have timeout in seconds
    // if timeout is reached, nullptr is returned
    server_task_result_ptr recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout);

    // single-task version of recv()
    server_task_result_ptr recv(int id_task);

    // Send a new result to a waiting id_task
    void send(server_task_result_ptr && result);

    // broadcast a new result to all waiting tasks
    // (used by router mode)
    void broadcast(server_task_result_ptr && result);

    // terminate the waiting loop
    void terminate();
};

// RAII wrapper to make working with server_queue and server_response easier
// it provides a generator-like API for server responses
// support pooling connection state and aggregating multiple results
struct server_response_reader {
    std::unordered_set<int> id_tasks;
    server_queue & queue_tasks;
    server_response & queue_results;
    size_t received_count = 0;
    bool cancelled = false;
    bool has_ctx_guard = false;
    int polling_interval_seconds;

    // tracking generation state and partial tool calls
    // only used by streaming completions
    std::vector<task_result_state> states;

    // should_stop function will be called each polling_interval_seconds
    server_response_reader(server_queue & queue_tasks, server_response & queue_results, int polling_interval_seconds)
        : queue_tasks(queue_tasks), queue_results(queue_results), polling_interval_seconds(polling_interval_seconds) {}
    ~server_response_reader() {
        stop();
    }

    int get_new_id() {
        return queue_tasks.get_new_id();
    }

    // pre-task ctx guard: held from create_response() until the tasks are
    // posted (or the reader dies), covering every model/vocab read on the
    // HTTP thread; released automatically by post_task/post_tasks/stop.
    // Returns false if a teardown hold outlasted hold_timeout_ms (see
    // server_queue::ctx_guard_enter).
    bool ctx_guard_acquire(int hold_timeout_ms = -1) {
        if (!has_ctx_guard) {
            if (!queue_tasks.ctx_guard_enter(hold_timeout_ms)) {
                return false;
            }
            has_ctx_guard = true;
        }
        return true;
    }
    void ctx_guard_release() {
        if (has_ctx_guard) {
            queue_tasks.ctx_guard_exit();
            has_ctx_guard = false;
        }
    }

    // if front = true, the task will be posted to the front of the queue (high priority)
    void post_task(server_task && task, bool front = false);
    void post_tasks(std::vector<server_task> && tasks, bool front = false);
    bool has_next() const;

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop);

    struct batch_response {
        bool is_terminated = false; // if true, indicates that processing was stopped before all results were received
        std::vector<server_task_result_ptr> results;
        server_task_result_ptr error; // nullptr if no error
    };
    // aggregate multiple results
    batch_response wait_for_all(const std::function<bool()> & should_stop);

    void stop();
};
