import os

def rmse(predictions, targets):
    return (((predictions - targets) ** 2).mean()) ** 0.5