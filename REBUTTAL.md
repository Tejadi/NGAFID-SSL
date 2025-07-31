# Rebuttal experiments

## 1. SimCLR + Classification head
| Task                             | Total Flights | Correct Flights | Accuracy |
|----------------------------------|---------------|------------------|----------|
| SimCLR Type Classification 10% of training dataset  | 1107          | 364              | 0.3288   |
| SimCLR Class Classification 10% of training dataset | 1107          | 328              | 0.2963   |
| SimCLR Type Classification 100% of training dataset | 1107          | 686              | 0.6197   |
| SimCLR Class Classification 100% of training dataset| 1107          | 327              | 0.2954   |
| SimCLR Type Classification 1% of training dataset   | 1107          | 780              | 0.7046   |
| SimCLR Class Classification 1% of training dataset  | 1107          | 389              | 0.3514   |

## 2. ConvMHSA Classification
| Task                             | Total Flights | Correct Flights | Accuracy |
|----------------------------------|---------------|------------------|----------|
| ConvMHSA Type Classification 100% of training dataset                  | 1107          | 1104             | 0.9973   |
| ConvMHSA Class Classification 100% of training dataset                 | 1107          | 1106              | 0.9991   |
| ConvMHSA Type Classification 10% of training dataset                   | 1107          | 1088             | 0.9828   |
| ConvMHSA Class Classification 10% of training dataset                  | 1107          | 447              | 0.4038   |
| ConvMHSA Type Classification 1% of training dataset                    | 1107          | 953              | 0.8609   |
| ConvMHSA Class Classification 1% of training dataset                   | 1107          | 377              | 0.3406   |
