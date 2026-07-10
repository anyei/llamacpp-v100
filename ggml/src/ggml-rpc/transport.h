#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();

// LAN presence beacon + discovery over UDP multicast (TASKS.md #29d).
// No auth — trusted private networks only, like the RPC protocol itself.
// group syntax: "ADDR:PORT"; nullptr/empty selects RPC_DISCOVER_GROUP_DEFAULT.
static constexpr const char * RPC_DISCOVER_GROUP_DEFAULT = "239.255.77.99:50153";

// worker side: detached thread multicasts make_payload() every ~2 s.
// iface_host selects the egress interface — pass the rpc-server bind host so the
// beacon never leaves the interface the RPC port itself is on ("0.0.0.0" => OS default).
bool rpc_announce_start(const char * group, const char * iface_host, std::function<std::string()> make_payload);

// coordinator side: listen on the group for timeout_ms; returns unique (src_ip, payload) pairs.
std::vector<std::pair<std::string, std::string>> rpc_discover_listen(const char * group, int timeout_ms);

// short non-blocking TCP connect probe — reachability check for discovered candidates
// (the OS-default blocking connect can stall for minutes on a firewalled host)
bool rpc_probe_endpoint(const char * host, int port, int timeout_ms);
