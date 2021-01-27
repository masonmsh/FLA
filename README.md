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
The datasets of NAIS and DeepICF are common, so only upload them in the NAIS folder.
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
We have added factor-level attention to the two methods: NAIS and DeepICF. The codes are in the corresponding folders.

**COMMON**
- **Batch_gen:** Randomly sort the data and train in batches.
- **Dataset:** Load and process data.
- **Evaluate:** Evaluate the experimental results: use two evaluation indicators, HR (Hit Ratio) and NDCG (Normalized Discounted Cumulative Gain).

**NAIS**
- **FISM:** A groundbreaking learning-based ICF (Item-based Collaborative Filtering) model, not only as a baseline, but also as a pre-processing for NAIS and DeepICF methods.
- **NAIS:** A method that considers the different impacts of historical items on the target item and uses the attention mechanism to model the item-level attention.
- **NAIS1:** Add factor-level attention mechanism to NAIS: use two-layer attention (corresponding to Design1 in the paper).
- **NAIS2:** Add factor-level attention mechanism to NAIS: use single-layer attention (corresponding to Design2 in the paper).

**DeepICF**
- **DeepICFa:** A deep ICF method that can capture high-level interactions between items.
- **DeepICF1:** Add factor-level attention mechanism to DeepICF: use two-layer attention.
- **DeepICF2:** Add factor-level attention mechanism to DeepICF: use single-layer attention.

## Examples to run the codes
Please open the terminal to execute in the code root directory. Both NAIS and DeepICF use FISM as pre-training; FLA uses the corresponding method as pre-training, that is, $FLA_{NAIS}$ uses NAIS and $FLA_{DICF}$ uses DeepICF.
### pretraining
#### Run FISM
```
cd NAIS
```
```
python FISM.py --dataset Digital_Music --lr 0.01 --embed_size 16
```
#### Run NAIS
```
cd NAIS
```
```
python NAIS.py --dataset Digital_Music --lr 0.01 --embed_size 16 --weight_size 16 --beta 0.7 --pretrain 1
```
#### Run DeepICF
```
cd DeepICF
```
```
python DeepICFa.py --dataset Digital_Music --lr 0.01 --embed_size 16 --weight_size 16 --layers [32,16,8] --beta 0.5 --pretrain 1
```
### Run FLA
#### Run $FLA_{NAIS}$
**without pretraining**
```
cd NAIS
```
*Design1*
```
python NAIS1.py--dataset Digital_Music --lr 0.01 --embed_size 16 --weight_size 16 --beta 0.7
```
*Design2*
```
python NAIS2.py--dataset Digital_Music --lr 0.01 --embed_size 16 --weight_size 16 --beta 0.7
```
**with pretraining**
```
cd NAIS
```
*Design1*
```
python NAIS1.py--dataset Digital_Music --lr 0.01 --embed_size 16 --weight_size 16 --beta 0.7 --pretrain 1
```
*Design2*
```
python NAIS2.py--dataset Digital_Music --lr 0.01 --embed_size 16 --weight_size 16 --beta 0.7 --pretrain 1
```
#### Run $FLA_{DICF}$
**without pretraining**
```
cd DeepICF
```
*Design1*
```
python DeepICF1.py--dataset Digital_Music --lr 0.01 --embed_size 16 --weight_size 16 --beta 0.5
```
*Design2*
```
python DeepICF2.py--dataset Digital_Music --lr 0.01 --embed_size 16 --weight_size 16 --beta 0.5
```
**with pretraining**
```
cd DeepICF
```
*Design1*
```
python DeepICF1.py--dataset Digital_Music --lr 0.01 --embed_size 16 --weight_size 16 --beta 0.5 --pretrain 1
```
*Design2*
```
python DeepICF2.py--dataset Digital_Music --lr 0.01 --embed_size 16 --weight_size 16 --beta 0.5 --pretrain 1
```