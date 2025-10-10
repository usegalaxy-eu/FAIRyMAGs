# Overview
When building models to predict resource estimation of HPC jobs its crucial to have evaluation methods that reflect the real world usage of such prediction systems. This can help us to fine-tune parameter, select best models, and avoid false sense of improvement.

In this notebook we explore different evaluation metrics and how they can be used to evaluate the performance of our models.



# 1. 

## Overview of what is this about
When building models to predict resource estimation of HPC jobs its crucial to have evaluation methods that reflect the real world usage of such prediction systems. This can help us to fine-tune parameter, select best models, and avoid false sense of improvement.

In this notebook we explore different evaluation metrics and how they can be used to evaluate the performance of our models.

* Start with the *problem, question, or assumption* everyone shares.
    - Problem: Selecting the best model for resource estimation is challenging as many metrics can be used to capture different aspects of model performance.
    - Question: How to evaluate models in a way that reflects real-world usage?
    - Assumption: Common performance metrics like MAE, RMSE, and R², as well as safety/efficiency bulk metric, are good approximations to select best model.


## Plots showing base metrics (MAE, MSE, R2) and safety/efficiency bulk metric for different models
| Include Gaussian Process method

## Our current allocation method, add it for comparison



    
![jpg](evaluation_metrics_files/evaluation_metrics_13_0.png)
    



    
![png](evaluation_metrics_files/evaluation_metrics_18_0.png)
    


## What are we seeing in the above plots?
- We see that based on the metrics, the Random Forest model performs best. It has better accuacy according to MAE, RMSE, and R². 
- Also using the trade-off plot, we can see that the Random Forest model also provides a better trade-off between underprediction and excess memory allocation, as it has a lowrd area under the curve.

So based on these results, the Random Forest model is the best model to use for predicting peak memory usage for metagenomic assemblies using the features provided in the dataset.


# 2. 





- but I explored the data a bit more, and I saw some interesting patterns

## Plots showing the different distribution of Reallife memusage frequency vs evaluation dataset


    
![png](evaluation_metrics_files/evaluation_metrics_23_0.png)
    


- We saw that the real pattern of memory usage in real life is different from the one in the evaluation dataset. Real life jobs tend to be smaller, and the evaluation dataset has a lot of larger jobs.
- This suggested that the classical evaluation metrics could be biased toward the peak_merory distribution of the validation dataset, therefore could not reflect the real world performance of the models.

So we decided to explore more, and see how each models performs across the entire distribution of peak_memory usage, and not only on average, or sum.

## Plots showing bins of mem vs underprediction failure rate


    
![png](evaluation_metrics_files/evaluation_metrics_28_0.png)
    


- Here we see that all models tend to have a larger failure rate on high mem jobs, and overallocation is greter in low mem jobs.
- Looking at this is not obvious wich method is better

# 🎯 3. 


* Connect the finding to **human or business implications**.
- Models low MAE, MSE, R2 do not necessary perform well for this problem as the metric doesnot dicern between underestimation, and overestimation. But in real life this means failure of jobs, and posibly resource duplication for retry jobs.
- Bulk waste/safety metric does not dicern between different job sizes, and biases the metric toward the distribution of evaluation dataset

* Interpret the *drivers* and *consequences* of the pattern.
- HPC administator are more concened about large mem jobs, as they have limited machines with high memories. 
- Users are more concerned about jobs failing due to underestimation, as they have to wait longer.

* Translate stats into **meaning**.
This means that model selection should be based on metrics that reflect real-world usage:
- Evaluation metrics on different job sizes should reflect the actual distribution of jobs requested (users send mostly some type of metagenomes i.e. gut)
- Evaluation metric should take into account the **Total cost for one job** which is also a function of the retry policy of the HPC system:
$$
C(y; a_0) = 
\underbrace{\sum_{r=0}^{R-1} C_{\text{fail}}(y, a_r)}_{\text{cost of retries}} 
+ 
\underbrace{C_{\text{over}}(y, a_R)}_{\text{waste on the successful attempt}}
$$




# 🔧 4.


## Plot that exemplifies how total waste of resources is a function of the retry policy, and an initial memory allocation closer to the true, could result in more waste of resources if underpredicts
- the figure tries to develop intuition on how MAE like metrics is not alway guarantee of low waste. The figure show for a kob with true mem 120, how different initial allocations results, in failed jobs and wasted memory. Each job in the attemp had height=memory, and lenght - time elapsed. The total area is the total cost of the job. The sum ob jobs reflect all the allocated resources to it.

    /tmp/ipykernel_1226469/3462535538.py:90: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
      plt.tight_layout(rect=(0, 0, 1, 0.98))



    
![png](evaluation_metrics_files/evaluation_metrics_35_1.png)
    


# 🚀 5. End with Action — “What Now?”

Great insights inspire movement. Don’t stop at the reveal.

* End with what decision or action this insight should drive.
- concrete action is to use the evaluation metrics that take into account:
    - the real world distribution of jobs
    - memory usage rate (memory*time), and not just peak memory
    - the total cost of one job including retries and waste, this involves using the retry policy of the HPC system
    - the final metric should be like a weighted cost function that takes into account the above points, and the cost acress 1000 jobs that reflect real world usage
* Suggest a clear next step or hypothesis to test.
- We have a library to produce stats and visualise the total time and memory usage of different models.         
> “By shifting our selection strategy, we could actualy select a better model.”

---

    | predictor                    |   waste_per_1000_mean |   waste_per_1000_lo |   waste_per_1000_hi |   wall_per_1000_mean |   wall_per_1000_lo |   wall_per_1000_hi |   failure_rate_mean |   failure_rate_lo |   failure_rate_hi |
    |:-----------------------------|----------------------:|--------------------:|--------------------:|---------------------:|-------------------:|-------------------:|--------------------:|------------------:|------------------:|
    | Heuristic default            |           3.27232e+06 |         2.40202e+06 |         4.46338e+06 |              50020.7 |            37389.7 |            67498   |           0.0367923 |             0.026 |             0.048 |
    | Random Forest (All) default  |           3.51789e+06 |         2.3217e+06  |         5.00405e+06 |              55830.2 |            43594.8 |            71602.9 |           0.267915  |             0.241 |             0.295 |
    | Gaussian Processes biome 0.9 |           1.96113e+06 |         1.51106e+06 |         2.49544e+06 |              41996.8 |            34628.2 |            51083   |           0.0867093 |             0.07  |             0.104 |
    | Random Forest (All) + 40GB   |           3.19231e+06 |         2.45718e+06 |         4.04184e+06 |              44411.7 |            35736.8 |            54714.6 |           0.031274  |             0.021 |             0.042 |
    | Dummy 34 GB                  |           5.33693e+06 |         3.31382e+06 |         7.9853e+06  |              87251.7 |            64616.1 |           117202   |           0.277125  |             0.249 |             0.306 |



    
![png](evaluation_metrics_files/evaluation_metrics_39_0.png)
    



    
![png](evaluation_metrics_files/evaluation_metrics_39_1.png)
    



    
![png](evaluation_metrics_files/evaluation_metrics_39_2.png)
    



    
![png](evaluation_metrics_files/evaluation_metrics_39_3.png)
    


- This results show that despite having a low MAE, MSE, R2, the ML models does not clearly beat the used heuristics in term of waste.
- If total thorughtput is seek to be optimised maybe ML models help.


Next steps:
- To gether more real world data to get a better estimate of the real world distribution of jobs
- imprive traings, using failed job logs could also help
- retrain using new dataset


