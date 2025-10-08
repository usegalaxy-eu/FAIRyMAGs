# Ideas

## Training
- Trainig objective function should have cost
# For evaluation
it should take into account:
- Time used memory
- The cost of failing is high (e.g., if we underpredict memory, the job will be killed, and we will have to resubmit it, which will take more time and resources)
- On failed jobs, at what time it failed?
- Threshold-weighted scoring rules: Emphasize performance beyond a size threshold 
y
>
y
0
y>y 
0
​	
  (e.g., large jobs). This is common in forecasting; it formalizes “care more about the tail.”