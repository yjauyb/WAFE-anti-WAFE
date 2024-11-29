import torch

class accuracy():
    """
    get topk accuracy over the k top predictions for the specified values of k.
    
    """
    def __init__(self, topk:tuple = (1,), device = None, num_classes = 1000) -> None:
        self.topk = topk
        self.num_topk_correct = [0]*len(topk)
        self.num_total_sample = 0 # total number of sample
        self.top1_correct_per_class = torch.zeros(num_classes, device=device)
        self.num_sample_per_class = torch.zeros(num_classes, device=device)
        self.num_classes = num_classes
    def step_accuray(self, predict:torch.Tensor, target:torch.Tensor) -> None: 
        """
        get number of correct predicted and number of sample in a step, use to calculate the accuracy at the end of evaluation.
        predict: predicted logits. shape (B, N_class)
        target: class label. Shape(B). Class label in Int.
        """       
        maxk = min(max(self.topk), predict.size()[1]) 
        self.num_total_sample += target.size(0)
        _, pred = predict.topk(k=maxk, dim=1, largest=True, sorted=True)  # get index of the topk (B, K)
        pred = pred.t() #(K, B)
        correct = pred.eq(target.reshape(1, -1).expand_as(pred)) #(K, B)
        for i in range(len(self.topk)):
            k = min(self.topk[i], maxk)
            num_correct = correct[:k].sum().item()
            self.num_topk_correct[i] += num_correct
        
        # count per class for top1 accuracy
        top1_correct = correct[0].to(self.top1_correct_per_class.dtype) #(B)        
        self.top1_correct_per_class.index_add_(dim=0, index=target, source=top1_correct) # (K)
        self.num_sample_per_class.add_(torch.nn.functional.one_hot(target, num_classes=self.num_classes).sum(dim=0)) # (K)

    def get_final_accuracy(self) -> list:
        """
        return the accuracy in percent.
        """
        return [round((num/self.num_total_sample)*100.0, 2) for num in self.num_topk_correct], self.top1_correct_per_class/self.num_sample_per_class
      