# Rebuttal experiments

## 1. SimCLR + Classification head
| Task                             | Total Flights | Correct Flights | Accuracy |
|----------------------------------|---------------|------------------|----------|
| simclr type classification 10%  | 1107          | 364              | 0.3288   |
| simclr class classification 10% | 1107          | 328              | 0.2963   |
| simclr type classification 100% | 1107          | 686              | 0.6197   |
| simclr class classification 100%| 1107          | 327              | 0.2954   |
| simclr type classification 1%   | 1107          | 780              | 0.7046   |
| simclr class classification 1%  | 1107          | 389              | 0.3514   |

## 2. ConvMHSA Classification
| Task                             | Total Flights | Correct Flights | Accuracy |
|----------------------------------|---------------|------------------|----------|
| mhsa type 100%                  | 1107          | 1104             | 0.9973   |
| mhsa class 100%                 | 1107          | 1106              | 0.9991   |
| mhsa type 10%                   | 1107          | 1088             | 0.9828   |
| mhsa class 10%                  | 1107          | 447              | 0.4038   |
| mhsa type 1%                    | 1107          | 953              | 0.8609   |
| mhsa class 1%                   | 1107          | 377              | 0.3406   |
