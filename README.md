Implementation of Mased Connditional Video Diffusion (MCVD) models. Using Denoised Diffusion model to conduct Video Predictionn. 

Currently, the model is trained on 20 minutes of scraped Youtube Videos of driving scenes in Tokyo and Kyoto city:
<img width="732" alt="image" src="https://github.com/fangyuan-ksgk/MCVD/assets/66006349/94a7f1bf-8903-46b3-a51f-515cc879126c">
The performance seems to be ok, but I feel like there are still bugs inside the codebase, for instance, the color of the predicted pic and the conditional pic are somehow different. 

```

--- Ongoing: 
YouTube Driving Dataset Video Prediction
* Downloaded Driving Video from Youtube , Train MCVD on these driving video to conduct prediction tasks
<img width="1358" alt="image" src="https://github.com/fangyuan-ksgk/MCVD/assets/66006349/6095e783-e3ae-4ad1-818c-e9ffb9b50c87">

* MCVD training pipeline completed -- but performance is still less than promising here
