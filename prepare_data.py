import copy

import pandas as pd

from utils.dataset_splitter import KFold
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

data_dir = Path("/kaggle/input")


def get_data(year):
    df = pd.read_csv(data_dir/f"birdclef-{year}/train_metadata.csv")
    df["data_year"] = year
    if year == 2021:
        df["filename"] = df.primary_label + "/" + df.filename
    return df

data = pd.concat(
    [get_data(i) for i in range(2021, 2024)]
)
# print(data.head())
data = data[["filename", "primary_label", "secondary_labels", "rating", "time", "data_year"]]

#t = copy.deepcopy(data)
#t = t[t["data_year"].isin([2021])]
# t = t[t["primary_label"].isin(["abethr", "abythr", "afghor1"])]
#print(t.sample(frac=1.0).head(100))


target_columns = "abethr1 abhori1 abythr1 afbfly1 afdfly1 afecuc1 affeag1 afgfly1 afghor1 afmdov1 afpfly1 afpkin1 afpwag1 afrgos1 afrgrp1 afrjac1 afrthr1 amesun2 augbuz1 bagwea1 barswa bawhor2 bawman1 bcbeat1 beasun2 bkctch1 bkfruw1 blacra1 blacuc1 blakit1 blaplo1 blbpuf2 blcapa2 blfbus1 blhgon1 blhher1 blksaw1 blnmou1 blnwea1 bltapa1 bltbar1 bltori1 blwlap1 brcale1 brcsta1 brctch1 brcwea1 brican1 brobab1 broman1 brosun1 brrwhe3 brtcha1 brubru1 brwwar1 bswdov1 btweye2 bubwar2 butapa1 cabgre1 carcha1 carwoo1 categr ccbeat1 chespa1 chewea1 chibat1 chtapa3 chucis1 cibwar1 cohmar1 colsun2 combul2 combuz1 comsan crefra2 crheag1 crohor1 darbar1 darter3 didcuc1 dotbar1 dutdov1 easmog1 eaywag1 edcsun3 egygoo equaka1 eswdov1 eubeat1 fatrav1 fatwid1 fislov1 fotdro5 gabgos2 gargan gbesta1 gnbcam2 gnhsun1 gobbun1 gobsta5 gobwea1 golher1 grbcam1 grccra1 grecor greegr grewoo2 grwpyt1 gryapa1 grywrw1 gybfis1 gycwar3 gyhbus1 gyhkin1 gyhneg1 gyhspa1 gytbar1 hadibi1 hamerk1 hartur1 helgui hipbab1 hoopoe huncis1 hunsun2 joygre1 kerspa2 klacuc1 kvbsun1 laudov1 lawgol lesmaw1 lessts1 libeat1 litegr litswi1 litwea1 loceag1 lotcor1 lotlap1 luebus1 mabeat1 macshr1 malkin1 marsto1 marsun2 mcptit1 meypar1 moccha1 mouwag1 ndcsun2 nobfly1 norbro1 norcro1 norfis1 norpuf1 nubwoo1 pabspa1 palfly2 palpri1 piecro1 piekin1 pitwhy purgre2 pygbat1 quailf1 ratcis1 raybar1 rbsrob1 rebfir2 rebhor1 reboxp1 reccor reccuc1 reedov1 refbar2 refcro1 reftin1 refwar2 rehblu1 rehwea1 reisee2 rerswa1 rewsta1 rindov rocmar2 rostur1 ruegls1 rufcha2 sacibi2 sccsun2 scrcha1 scthon1 shesta1 sichor1 sincis1 slbgre1 slcbou1 sltnig1 sobfly1 somgre1 somtit4 soucit1 soufis1 spemou2 spepig1 spewea1 spfbar1 spfwea1 spmthr1 spwlap1 squher1 strher strsee1 stusta1 subbus1 supsta1 tacsun1 tafpri1 tamdov1 thrnig1 trobou1 varsun2 vibsta2 vilwea1 vimwea1 walsta1 wbgbir1 wbrcha2 wbswea1 wfbeat1 whbcan1 whbcou1 whbcro2 whbtit5 whbwea1 whbwhe3 whcpri2 whctur2 wheslf1 whhsaw1 whihel1 whrshr1 witswa1 wlwwar wookin1 woosan wtbeat1 yebapa1 yebbar1 yebduc1 yebere1 yebgre1 yebsto1 yeccan1 yefcan yelbis1 yenspu1 yertin1 yesbar1 yespet1 yetgre1 yewgre1".split()
# target_columns = "hawcre houfin".split()

data = data[data["primary_label"].isin(target_columns)]


import librosa
def get_duration(row):
    path = data_dir/f"birdclef-{row.data_year}/{'train_audio' if row.data_year in [2022, 2023] else 'train_short_audio'}/{row.filename}"
    # Load the audio file
    sr = 32000
    audio_file, sr = librosa.load(path, sr=sr)
    # Get the duration in milliseconds
    duration_ms =len(audio_file)
    del audio_file
    return duration_ms

# data = data.sample(frac=1.0).head(100)
data["duration"] = data.progress_apply(get_duration, axis=1)
# print(data.head())
data["duration"] = data["duration"].astype("int")

from utils.dataset_splitter import KFold
kfold = KFold(random_seed=42, k_folds=5)
data = kfold.group_split(data, "primary_label")
print(data.head(30))

data.to_csv("folds.csv", index=False)
