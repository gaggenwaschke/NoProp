# log for initial try

## MNIST resnet18 backbone  with 100 epoch
```
--- resnet18 on cuda ---
Epoch 01 loss 14.9257
Epoch 02 loss 1.6941
Epoch 03 loss 1.1895
Epoch 04 loss 0.9984
Epoch 05 loss 1.0117 | Acc 96.41%
Epoch 06 loss 1.0015
Epoch 07 loss 1.0154
Epoch 08 loss 0.8901
Epoch 09 loss 0.8856
Epoch 10 loss 0.8534 | Acc 96.64%
Epoch 11 loss 0.8452
Epoch 12 loss 0.8399
Epoch 13 loss 0.7876
Epoch 14 loss 0.7591
Epoch 15 loss 0.9294 | Acc 92.26%
Epoch 16 loss 0.7894
Epoch 17 loss 0.7055
Epoch 18 loss 0.7082
Epoch 19 loss 0.6623
Epoch 20 loss 0.6917 | Acc 97.49%
Epoch 21 loss 0.6411
Epoch 22 loss 0.6503
Epoch 23 loss 0.6236
Epoch 24 loss 0.5796
Epoch 25 loss 0.5703 | Acc 98.19%
Epoch 26 loss 0.5843
Epoch 27 loss 0.5962
Epoch 28 loss 0.5654
Epoch 29 loss 0.6034
Epoch 30 loss 0.5708 | Acc 98.18%
Epoch 31 loss 0.5551
Epoch 32 loss 0.5536
Epoch 33 loss 0.5590
Epoch 34 loss 0.5706
Epoch 35 loss 0.6444 | Acc 96.35%
Epoch 36 loss 0.5946
Epoch 37 loss 0.6180
Epoch 38 loss 0.6055
Epoch 39 loss 0.5820
Epoch 40 loss 0.5500 | Acc 98.53%
Epoch 41 loss 0.5375
Epoch 42 loss 0.5484
Epoch 43 loss 0.5507
Epoch 44 loss 0.5361
Epoch 45 loss 0.5389 | Acc 98.84%
Epoch 46 loss 0.5263
Epoch 47 loss 0.5201
Epoch 48 loss 0.5160
Epoch 49 loss 0.5183
Epoch 50 loss 0.5227 | Acc 98.88%
Epoch 51 loss 0.5487
Epoch 52 loss 0.6648
Epoch 53 loss 0.7750
Epoch 54 loss 0.9009
Epoch 55 loss 0.8544 | Acc 91.85%
Epoch 56 loss 0.8039
Epoch 57 loss 0.6627
Epoch 58 loss 0.5831
Epoch 59 loss 0.5568
Epoch 60 loss 0.5505 | Acc 98.42%
Epoch 61 loss 0.5477
Epoch 62 loss 0.5404
Epoch 63 loss 0.5435
Epoch 64 loss 0.5424
Epoch 65 loss 0.5316 | Acc 99.00%
Epoch 66 loss 0.5365
Epoch 67 loss 0.5620
Epoch 68 loss 0.5350
Epoch 69 loss 0.5273
Epoch 70 loss 0.5765 | Acc 97.92%
Epoch 71 loss 0.5387
Epoch 72 loss 0.5280
Epoch 73 loss 0.5347
Epoch 74 loss 0.5471
Epoch 75 loss 0.6241 | Acc 87.31%
Epoch 76 loss 0.6469
Epoch 77 loss 0.6145
Epoch 78 loss 0.6364
Epoch 79 loss 0.5708
Epoch 80 loss 0.5830 | Acc 98.59%
Epoch 81 loss 0.5636
Epoch 82 loss 0.5423
Epoch 83 loss 0.5856
Epoch 84 loss 0.5509
Epoch 85 loss 0.5450 | Acc 98.25%
Epoch 86 loss 0.5400
Epoch 87 loss 0.5234
Epoch 88 loss 0.5179
Epoch 89 loss 0.5136
Epoch 90 loss 0.5129 | Acc 99.13%
Epoch 91 loss 0.5236
Epoch 92 loss 0.5114
Epoch 93 loss 0.5100
Epoch 94 loss 0.5043
Epoch 95 loss 0.5066 | Acc 99.21%
Epoch 96 loss 0.5041
Epoch 97 loss 0.5034
Epoch 98 loss 0.5048
Epoch 99 loss 0.5025
Epoch 100 loss 0.5020 | Acc 99.22%

Final Heun multi-T eval:
Heun T=  2 acc 52.4300%
Heun T=  5 acc 96.0500%
Heun T= 10 acc 2.3500%
Heun T= 20 acc 9.1800%
Heun T= 30 acc 99.2800%
Heun T= 40 acc 99.2900%
Heun T= 50 acc 99.3000%
Heun T= 60 acc 99.2500%
Heun T= 70 acc 99.2400%
Heun T= 80 acc 99.2500%
Heun T= 90 acc 99.2300%
Heun T=100 acc 99.2600%
Heun T=200 acc 99.2800%

Final Euler eval:
Euler T=  2 acc 99.2700%
Euler T=  5 acc 92.2300%
Euler T= 10 acc 74.9200%
Euler T= 20 acc 99.2300%
Euler T= 30 acc 99.2600%
Euler T= 40 acc 99.2500%
Euler T= 50 acc 99.2500%
Euler T= 60 acc 99.2300%
Euler T= 70 acc 99.3200%
Euler T= 80 acc 99.2900%
Euler T= 90 acc 99.2900%
Euler T=100 acc 99.2700%
Euler T=200 acc 99.2900%

```

## MNIST resnet50 backbone with 100 epoch
```

--- resnet50 on cuda ---
Epoch 01 loss 25.7987
Epoch 02 loss 1.8336
Epoch 03 loss 1.4491
Epoch 04 loss 1.1448
Epoch 05 loss 1.0820 | Acc 91.50%
Epoch 06 loss 1.0243
Epoch 07 loss 1.1611
Epoch 08 loss 1.0712
Epoch 09 loss 0.8863
Epoch 10 loss 0.9041 | Acc 90.44%
Epoch 11 loss 0.8612
Epoch 12 loss 0.8083
Epoch 13 loss 0.7434
Epoch 14 loss 0.7297
Epoch 15 loss 0.8067 | Acc 94.58%
Epoch 16 loss 0.7424
Epoch 17 loss 0.7118
Epoch 18 loss 0.6183
Epoch 19 loss 0.5836
Epoch 20 loss 0.5813 | Acc 97.41%
Epoch 21 loss 0.5726
Epoch 22 loss 0.5689
Epoch 23 loss 0.6009
Epoch 24 loss 0.5764
Epoch 25 loss 0.5639 | Acc 97.49%
Epoch 26 loss 0.5550
Epoch 27 loss 0.5360
Epoch 28 loss 0.5345
Epoch 29 loss 0.5296
Epoch 30 loss 0.5315 | Acc 97.77%
Epoch 31 loss 0.5507
Epoch 32 loss 0.6621
Epoch 33 loss 0.7016
Epoch 34 loss 0.6803
Epoch 35 loss 0.5994 | Acc 97.23%
Epoch 36 loss 0.5720
Epoch 37 loss 0.5506
Epoch 38 loss 0.5392
Epoch 39 loss 0.5545
Epoch 40 loss 0.5311 | Acc 98.05%
Epoch 41 loss 0.5240
Epoch 42 loss 0.5213
Epoch 43 loss 0.5153
Epoch 44 loss 0.5107
Epoch 45 loss 0.5052 | Acc 98.29%
Epoch 46 loss 0.5036
Epoch 47 loss 0.5035
Epoch 48 loss 0.5052
Epoch 49 loss 0.5561
Epoch 50 loss 0.6206 | Acc 96.60%
Epoch 51 loss 0.5928
Epoch 52 loss 0.5969
Epoch 53 loss 0.5775
Epoch 54 loss 0.5611
Epoch 55 loss 0.5334 | Acc 98.16%
Epoch 56 loss 0.5423
Epoch 57 loss 0.5279
Epoch 58 loss 0.5457
Epoch 59 loss 0.5291
Epoch 60 loss 0.5216 | Acc 98.24%
Epoch 61 loss 0.5227
Epoch 62 loss 0.5119
Epoch 63 loss 0.5138
Epoch 64 loss 0.5122
Epoch 65 loss 0.5055 | Acc 98.49%
Epoch 66 loss 0.5140
Epoch 67 loss 0.5382
Epoch 68 loss 0.5269
Epoch 69 loss 0.5378
Epoch 70 loss 0.5207 | Acc 98.21%
Epoch 71 loss 0.5282
Epoch 72 loss 0.6188
Epoch 73 loss 0.5527
Epoch 74 loss 0.5344
Epoch 75 loss 0.5777 | Acc 97.61%
Epoch 76 loss 0.5442
Epoch 77 loss 0.5526
Epoch 78 loss 0.5195
Epoch 79 loss 0.5140
Epoch 80 loss 0.5201 | Acc 98.60%
Epoch 81 loss 0.5212
Epoch 82 loss 0.5171
Epoch 83 loss 0.5104
Epoch 84 loss 0.5039
Epoch 85 loss 0.5091 | Acc 98.66%
Epoch 86 loss 0.5022
Epoch 87 loss 0.5034
Epoch 88 loss 0.5041
Epoch 89 loss 0.5042
Epoch 90 loss 0.5037 | Acc 98.63%
Epoch 91 loss 0.5008
Epoch 92 loss 0.5024
Epoch 93 loss 0.4992
Epoch 94 loss 0.4982
Epoch 95 loss 0.5002 | Acc 98.88%
Epoch 96 loss 0.4974
Epoch 97 loss 0.4969
Epoch 98 loss 0.4965
Epoch 99 loss 0.4969
Epoch 100 loss 0.5106 | Acc 98.72%

Final Heun multi-T eval:
Heun T=  2 acc 2.5100%
Heun T=  5 acc 1.7200%
Heun T= 10 acc 1.4300%
Heun T= 20 acc 2.6600%
Heun T= 30 acc 97.0000%
Heun T= 40 acc 98.8000%
Heun T= 50 acc 98.7300%
Heun T= 60 acc 98.7400%
Heun T= 70 acc 98.6700%
Heun T= 80 acc 98.6800%
Heun T= 90 acc 98.7100%
Heun T=100 acc 98.7700%
Heun T=200 acc 98.6900%

Final Euler eval:
Euler T=  2 acc 98.6400%
Euler T=  5 acc 36.8900%
Euler T= 10 acc 82.1200%
Euler T= 20 acc 98.7900%
Euler T= 30 acc 98.6600%
Euler T= 40 acc 98.7200%
Euler T= 50 acc 98.7200%
Euler T= 60 acc 98.7400%
Euler T= 70 acc 98.7000%
Euler T= 80 acc 98.7100%
Euler T= 90 acc 98.6400%
Euler T=100 acc 98.7100%
Euler T=200 acc 98.7300%
```
