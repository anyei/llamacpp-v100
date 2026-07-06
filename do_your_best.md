Alright, thank you for that, it was very good. Now i have a next round of improvements, this time i'm focusing on getting into the source code itself. Look for anything you
  think we can improve, prefill stage, token generation, multi gpu, it'll be your call, consult with me whenever you want me to take a decision but keep in mind our goal:
  1- Generation quality, should be minimum the same as we are having now, a lot better is allowed
  2- improving prefill stage and prompt caching is a plus
  3- token generation speed, focuse very well on this one, i want you to dig here, do your best, improve, simplify, fix what you can to make it the best
  4- any gap you want to fill, bug fixing, important missing feature is allowed but dont go too far, prioritize or ask for my decision
  5- kv cache compression area, see if we can implement the best compression in the market (turboquant perhaps from google's, or what you find is the best compressing gain while keeping 100% accuracy no loss compared to the mainstream quality)
  6 - draft a document with an analyse around true distributed inference and how to update the code to achieve it: 1 coordinator docker instance, n worker docker instances,
  each instance (coordinator and workers) may run on different gpu so that true parallel work can be accomplish hence increasing the generation and probably thruput when more
  than 2 gpus are available. we will follow up on this one but for now a documentation .md file is enough for you to pickup later and implement.
  This is hands on code, do your best here to make this a better product for me to run models with best generation quality possible and as fast generation as we can, worth to
  mention in the future i will totally use 4 nvidia tesla v100 so keep that in mind, my setup will not be restricted by two v100 gpus.
