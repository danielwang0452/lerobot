1. Task & Dataset 

Visualise dataset with
python -m lerobot.scripts.visualize_dataset     --repo-id lerobot/xarm_lift_medium     --episode-index 0

I will be using the xarm-lift-medium dataset from https://huggingface.co/datasets/lerobot/xarm_lift_medium, 
since it uses a simulated environment (required for eval) and has not been used in 
the provided pretrained models.The task for the arm is to simply pick up the cube and raise
it above a certain height by controlling 4 paramters: [x, y, z] for the position of the 
end effector and [w] for the gripper. The observation space is the 4D vector representing the agent's state 
and an 84 x 84 image. When I used the dataset visualisation tool, I found that, 
surprisingly, the robot attained a low reward and failed the task in every episode that I checked.
(see the recording). In practice, having some failure cases in the dataset can help
mitigate the distributional shift problem by essentially showing the policy how to correct
its mistakes, but in this case I did not find a single successful trajectory. This means that performing imitation learning alone on this dataset 
will not learn the task. However, since learning the task is not the focus of this exercise, I will proceed
with this dataset anyway.

2. Train a diffusion policy from scratch using scripts.train
python -m lerobot.scripts.train \
    --policy.type=diffusion \
    --env.type=xarm \
    --dataset.repo_id=lerobot/xarm_lift_medium \
    --wandb.enable=true \
    --policy.device=mps \
    --policy.push_to_hub=false \
    --eval_freq=1000 \
    --steps=10000 \
    --output_dir=part_b/outputs/train/xarm

For this part I trained the diffusion policy for 10 000 steps (much fewer than
the default 100 000) with evals every 1000 steps. 
![Alt text](WB_1.png)

Although eval loss is not tracked, I think it would be best to track eval loss 
as the policy is trained on the dataset (by splitting the dataset into
train/eval as we would do
in traditional supervised learning, which imitation learning essentially is). 
This way, we can make sure that the policy is not overfitting and we can 
distinguish between two cases if the model is not performing well in the environment: 1) If 
the model performs poorly but it has fit the 
dataset well with low eval loss, then the issue is not with the model itself
but possibly the expert demonstrations or environment 2) If the model performs poorly 
and we see that it has overfitted the dataset
with high eval loss, then we know this needs to be addressed first.

