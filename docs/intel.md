in game.py lines 652-653 were commented out so that the agent (even dead) is allowed an action.
This in addition of line 15,31-34 in headless_client.py which allows for another tick after the is_over condition is true allowing our agent to save the weights and config file.

This was necessary because the environment had no permanence and so we needed a way to to save the information(reward and trained weights) through a json config file between episodes so that our nn could learn.

