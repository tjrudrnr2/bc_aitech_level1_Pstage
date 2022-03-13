# bc_aitech_level1_Pstage

## Task
Classification mask, age, gender

## Directory
```
├── data/      ## data는 저작권 문제로 미포함
|   ├── image/
|   |   ├── train/ 
|   |   └── eval/
├── src/
|   ├── dataset.py
|   ├── evaluation.py
|   ├── imbalance.py
|   ├── inference.py
|   ├── loss.py
|   ├── model.py
|   ├── train.py
|   ├── Utils.py
├── output_csv/
|   ├── age_exp_output.csv
|   ├── gender_exp_output.csv
|   ├── mask_exp_output.csv
|   ├── total_exp_output.csv
```

## Data
* 전체 사람 수 : 4500명 (train : 2700 | eval : 1800)
* 한 사람당 사진의 개수 : 7 [마스크 5장, 이상하게 착용(코스크, 턱스크...) 1장, 미착용 1장]
* 전체 이미지 수 : 31500장 (train : 18900 | eval : 12600)
* 나이 : 20대 - 70대
* 성별 : 남,여
* 이미지 크기 : (384,512)
* classes_num=18
![131881060-c6d16a84-1138-4a28-b273-418ea487548d](https://user-images.githubusercontent.com/70709889/158053322-1088f359-371b-4688-865f-6296f159d95a.png)
