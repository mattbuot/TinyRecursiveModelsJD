# Training a Tiny Recursive Network (TRM) using Jacobian Descent on ARC-AGI 2

The goal of this repo is to explore the potential of combining the proven successes from TRM on ARC-AGI 2 with the general multi objective optimization framework enabled by JD.

TRM is a recursive architecture inspired by HRM [paper](https://arxiv.org/abs/2110.13711) [code](https://github.com/sapientinc/HRM) which simplifies some design choices and performs better on the selected benchmarks. These two model architectures have made a big impact because of how few trainable parameters they use (respectively 7M and 27M).

JD is a framework for training effectively on multiple objectives, it is implemented by the library TorchJD which integrates conveniently with PyTorch and enables various number of setups. The paper proposes a preferred strategy called UPGrad which can be intuitively described as follows: *at every parameter update: take a step that guarantees no objective will be worsened will making general improvement*.

## Setup 1: Splitting the token prediction error and the halting penalty

TRM trains implicitly on two objectives: predicting the right output grid and knowing when to stop the supervisions process at training time. In the original implementation these two objectives are combined using a weighted sum but it is worth trying to split them into two objectives and optimizing each independently. This way the learning from one wouldn't harm the other objective.

This is implemented by the **lm_loss_vs_q_halt_loss** option. The experiments show that there is not much conflict between these two loss functions hence the overhead of JD doesn't result in a very different performance.

## Setup 2: Instance-Wise Risk Minization (IWRM)

This approach proposed in Section 3 of the paper considers each training sample as a distinct objective. This means that when training on a batch of tasks, we have the guarantee to make improvement on **all** tasks at the same time. This could potentially prevent the model from learning wrong shortcuts because if a reasoning path is correct, then it should help predicting the output for **all output grids**.

This is implemented by the **iwrm_q_halt_in** option. We can see that the cosine similary with a traditional Gradient Descent update is very close to 1 meaning that there is not much difference between the two methods.

![image](assets/wandb/gd_sim_iwrm.png)


## Setup 3: Pixel-Wise Split

This approach is novel and consists of splitting the losses by every single grid pixel that is predicted. Rather than summing the error over each predicted token we treat each as a separate objective which can result in 900 different losses for a 30x30 output grid. Since this corresponds to a huge overhead we also propose to apply a random grouping that will add up the errors into for example 10 different buckets and only run UPGrad on these 10 combined losses. Intuitively this should force the model to learn patterns that help predict all pixels at once. For instance the model should not learn to predict a grid that is just the background color because even if this will be right for most pixels it is clearly not accurate for all pixels in general.

This is implemented by the **pixelwise_q_halt_in** option. Again, the similarity is almost always 1 except for a few case where it goes down to ~0.3 but this is so rare that it doesn't change much in the final results.

![image](assets/wandb/gd_sim_pixelwise.png)


## Setup 4: Refinements Split

This approach seemed like the most promising because of how nicely it integrates with the specificities of recurrent architectures. Since these models are trained to iterately improve their outputs, a meaningful choice is to consider each improvement step as a separate objective. This way the model should learn to make improvements after each refinement cycle in a way that will improve every step of the process. As explained in the [ablation study of HRM](https://arcprize.org/blog/hrm-analysis) by the ARC prize foundation, the refinement loop is a big driver of performance. Going from one to two cycles helps the performance jump from 19% to 32% on ARC-AGI 1. The gains are lower for higher cycles but this could potentially be explained by the conflicts in between improving different refinement step. The hope would then be that UPGrad could resolve this and then enable the model to leverage these cycles at their full potential.

Since TRM uses recurrence at multiple levels there are different possibilities about where to place the loss split. We implemented one at the higher level supervision process and one at the internal "reasoning" level. These two options can selected with the options **stack_supervisions_and_sum** and **stack_internal_and_sum**.

Stacking internal losses was very similar to GD at first but seemed to start diverging after quite some training time. It could be worth testing what happens afterwards...

![image](assets/wandb/gd_sim_stack_internal.png)

Stacking supervisions is quite memory intense since we need to store several full computation graphs so we limited to 4 supervisions steps maximum. It clearly produces a different result compared to GD but in most situations seemed to plateau and didn't reach more than 3% solves on ARC-AGI 2 evaluation set.

![image](assets/wandb/gd_sim_stack_supervisions.png)

## Setup 5: Hybrid approaches

Splitting methods can be combined together for example **iwrm_pixelwise_q_halt_in** or **stack_supervisions_and_iwrm**. The latter produces some interesting results as it seems to help preventing the model from giving up on more difficult tasks. I've encountered some situations where the model collapses and only predicts the tasks that it is able to solve and produce empty grids for the rest. IWRM helps overcoming that.

![image](assets/wandb/gd_sim_stack_supervisions_and_iwrm.png)

Here is a link to a W&B report demonstrating one of the best training run: [report](https://api.wandb.ai/links/kowadis/0fxss92i).

## Other attempts

1. Adding dropout didn't yield any significant improvement in generalization


2. Switching the loss to a differential loss produced some interesting results, overall performance was close to the original version but the model was sometime solving different puzzles which is worth mentioning. The goal was to guide the model more towards refining its previous outputs rather than producing the right answer at every shot.

$$differential\_loss = loss(logits_i - logits_{i-1})$$

3. Perturbation rate: the idea was to add some noise in between every refinement step at training time to perform a more robust reasoning trace. This feature still has some potential and requires further testing.
