import torch
import torch.nn.functional as F
import csv


def mean_vector_cal(feature_out, nway, kshot):
    """
    Args:
        feature_out : torch.tensor, features of data shot, [args.nway*args.kshot, embedding_dim]

    Returns:
        mean value of feature_out's same class, [args.nway, embedding_dim]
    """
    
    return torch.mean(feature_out.view(nway, kshot, -1), dim=1)

def class_vector_distance_softmax_loss(logits, nway=5, kshot=5, nquery=20):

    # loss = torch.zeros(1).cuda()

    logsoftmax_logtis = -F.log_softmax(-logits, dim=1).view(5, 4, 5)

    # for way in range(nway):
    #     loss += torch.sum(logsoftmax_logtis[way,:,way]) / 20
    #     print(torch.sum(logsoftmax_logtis[way,:,way]) / 20)

    loss = (torch.sum(logsoftmax_logtis[0,:,0]) + torch.sum(logsoftmax_logtis[1,:,1]) + \
            torch.sum(logsoftmax_logtis[2,:,2]) + torch.sum(logsoftmax_logtis[3,:,3]) + \
            torch.sum(logsoftmax_logtis[4,:,4])) / 20

    return loss

def square_euclidean_metric(a, b):
    """ Measure the euclidean distance (optional)
    Args:
        a : torch.tensor, features of data query
        b : torch.tensor, mean features of data shots or embedding features

    Returns:
        A torch.tensor, the minus euclidean distance
        between a and b
    """

    n = a.shape[0]
    m = b.shape[0]

    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)

    logits = torch.pow(a - b, 2).sum(2)
    return logits


def count_acc(logits, label):
    """ In each query set, the index with the highest probability or lowest distance is determined
    Args:
        logits : torch.tensor, distance or probabilty
        label : ground truth

    Returns:
        float, mean of accuracy
    """

    # when logits is distance
    pred = torch.argmin(logits, dim=1)

    # when logits is prob
    #pred = torch.argmax(logits, dim=1)

    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

def count_acc_transformer(logits, label):
    """ In each query set, the index with the highest probability or lowest distance is determined
    Args:
        logits : torch.tensor, distance or probabilty
        label : ground truth

    Returns:
        float, mean of accuracy
    """

    # when logits is distance
    # pred = torch.argmin(logits, dim=1)

    # when logits is prob
    pred = torch.argmax(logits, dim=1)

    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Averager():
    """ During training, update the average of any values.
    Returns:
        float, the average value
    """

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class csv_write():

    def __init__(self, args):
        self.f = open('StudentID_Name.csv', 'w', newline='')
        self.write_number = 1
        self.wr = csv.writer(self.f)
        self.wr.writerow(['id', 'prediction'])
        self.query_num = args.query

    def add(self, prediction):

        for i in range(self.query_num):
          self.wr.writerow([self.write_number, int(prediction[i].item())])
          self.write_number += 1

    def close(self):
        self.f.close()