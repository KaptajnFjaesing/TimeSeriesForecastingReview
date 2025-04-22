# Notes:
- We should investigate why some of the models are relatively poor in performance. This indicates some kind of error. Suspecius models include:
    + TFT darts
    + Light GBM w. sklear (why would the model become worse with a grid search? Because of bad feature engineering? Why have we used a distinctly bad feature engineering for this? xD)
    + why is Naive darts not closer to 1 at the first timestep in the horizon? It should be according to the definition of the MASE. This is unless there is a systematic where the average gradient in data grows over time, but is this the case?
- We should check all the equations in the models section in particular. Do they make sense? Is the text easy to understand? Does it contribute to a new reader? We could delete the section also? Or tweak it? I think it is good for us in the repo to remember what we have done, but perhaps we do not need it in the article?
- Is anything missing ? Feel free to add!