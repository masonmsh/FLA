Factor-level Attentive ICF for Recommendation
=
This is our official implementation for the paper:

**Factor-level Attentive ICF for Recommendation**   
Zhiyong Cheng, Shenghan Mei, Yangyang Guo, Lei Zhu, Liqiang Nie

Two designs based on factor-level attention: **Design1** & **Design2**. Design1 represents a two-layer network, and Design2 represents a single-layer network.

## Environment
- Python: '2.7'
- TensorFlow: '1.5.0'

## Dataset
The datasets of NAIS and DeepICF.
**train.rating:**
- Train file.
- Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

**test.rating:**
- Test file (positive instances).
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

**test.negative:**
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...

## Code file
We have added factor-level attention to the two methods: [NAIS](https://github.com/hexiangnan/Neural-Attentive-Item-Similarity-Model "NAIS") and [DeepICF](https://github.com/linzh92/DeepICF "DeepICF"). The codes are in the corresponding folders.

**COMMON**
- **Batch_gen:** Randomly sort the data and train in batches.
- **Dataset:** Load and process data.
- **Evaluate:** Evaluate the experimental results: use two evaluation indicators, HR (Hit Ratio) and NDCG (Normalized Discounted Cumulative Gain).

**NAIS**
- **NAIS1:** Add factor-level attention mechanism to NAIS: use two-layer attention (corresponding to Design1 in the paper).
- **NAIS2:** Add factor-level attention mechanism to NAIS: use single-layer attention (corresponding to Design2 in the paper).

**DeepICF**
- **DeepICF1:** Add factor-level attention mechanism to DeepICF: use two-layer attention.
- **DeepICF2:** Add factor-level attention mechanism to DeepICF: use single-layer attention.

*The source code links of NAIS and DeepICF:*  
[**[NAIS]**](https://github.com/hexiangnan/Neural-Attentive-Item-Similarity-Model "NAIS")
https://github.com/hexiangnan/Neural-Attentive-Item-Similarity-Model  
[**[DeepICF]**](https://github.com/linzh92/DeepICF "DeepICF")
https://github.com/linzh92/DeepICF

## Examples to run the codes
Please open the terminal to execute in the code root directory. Both NAIS and DeepICF use FISM as pre-training; FLA uses the corresponding method as pre-training, that is, $FLA_{NAIS}$ uses NAIS and $FLA_{DICF}$ uses DeepICF.

### Run $FLA_{NAIS}$
**without pretraining**
```
cd NAIS
```
*Design1*
```
python NAIS1.py --dataset Digital_Music --lr 0.001 --embed_size 16 --weight_size 16 --beta 0.7 --pretrain 0
```
*Design2*
```
python NAIS2.py --dataset Digital_Music --lr 0.001 --embed_size 16 --weight_size 16 --beta 0.7 --pretrain 0
```
**with pretraining**
```
cd NAIS
```
*Design1*
```
python NAIS1.py --dataset Digital_Music --lr 0.001 --embed_size 16 --weight_size 16 --beta 0.7 --pretrain 1
```
*Design2*
```
python NAIS2.py --dataset Digital_Music --lr 0.001 --embed_size 16 --weight_size 16 --beta 0.7 --pretrain 1
```
### Run $FLA_{DICF}$
**without pretraining**
```
cd DeepICF
```
*Design1*
```
python DeepICF1.py --dataset Digital_Music --lr 0.001 --embed_size 16 --weight_size 16 --beta 0.5 --pretrain 0
```
*Design2*
```
python DeepICF2.py --dataset Digital_Music --lr 0.001 --embed_size 16 --weight_size 16 --beta 0.5 --pretrain 0
```
**with pretraining**
```
cd DeepICF
```
*Design1*
```
python DeepICF1.py --dataset Digital_Music --lr 0.001 --embed_size 16 --weight_size 16 --beta 0.5 --pretrain 1
```
*Design2*
```
python DeepICF2.py --dataset Digital_Music --lr 0.001 --embed_size 16 --weight_size 16 --beta 0.5 --pretrain 1
```