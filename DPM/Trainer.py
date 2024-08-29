class IncrimentalTraining:
    def __init__(self,param1,param2) -> None:
        self.param1=param1
        self.param2=param2
    def retrain(self,model,dataset,freq):
        for data in dataset:
            update(model,data,freq)
        return model
    