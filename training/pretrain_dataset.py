import math
import os
import ast
import librosa

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
    ):
        ignore_labels = ('akiapo', 'aniani', 'apapan', 'barpet', 'crehon', 'elepai', 'ercfra',
                         'hawama', 'hawcre', 'hawgoo', 'hawhaw', 'hawpet1', 'houfin', 'iiwi',
                         'jabwar', 'maupar', 'omao', 'puaioh', 'skylar', 'warwhe1', 'yefcan')
        df2021 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2021/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2022 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2022/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2023 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2023/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2021["data_year"] = 2021
        df2022["data_year"] = 2022
        df = pd.concat([df2021, df2022], ignore_index=True)
        ##df2023["data_year"] = 2023
        # df = pd.concat([df2021, df2022, df2023], ignore_index=True)

        print(len(df))
        df = df[~df.primary_label.isin(ignore_labels)]
        print(len(df))
        labels = list(set(df.primary_label.unique()))
        labels.sort()
        self.labels = labels
        self.df = df
        self.dataset_dir = dataset_dir


        self.duration = 15
        self.sr = 32000
        self.dsr = self.duration * self.sr
        self.bird2id = {x: idx for idx, x in enumerate(labels)}

    def load_one(self, filename, offset, duration):
        try:
            wav, _ = librosa.load(filename, sr=None, offset=offset, duration=duration)
        except:
            print("failed reading", filename)
        return wav


    def __getitem__(self, i):
        row = self.df.iloc[i]
        data_year = row['data_year']
        filename = os.path.join(self.dataset_dir, f"birdclef-{data_year}",
                                "train_audio" if data_year != 2021 else "train_short_audio", row["primary_label"], row['filename'].split("/")[-1])

        ## wav

        wav_len_sec = librosa.get_duration(filename=filename, sr=None)
        duration = self.duration
        max_offset = wav_len_sec - duration
        max_offset = max(max_offset, 1)
        offset = np.random.randint(max_offset)


        wav = self.load_one(filename, offset=offset, duration=self.duration)
        if wav.shape[0] < (self.dsr):
            wav = np.pad(wav, (0, self.dsr - wav.shape[0]))

        ## labels
        labels = torch.zeros((len(self.labels),))
        labels[self.bird2id[row['primary_label']]] = 1.0
        for x in ast.literal_eval(row['secondary_labels']):
            try:
                labels[self.bird2id[x]] = 1.0
            except:
                continue

        return {
            "wav": torch.tensor(wav).unsqueeze(0),
            "labels": labels,
        }

    def __len__(self):
        return len(self.df)

target_columns = "abethr1 abhori1 abythr1 afbfly1 afdfly1 afecuc1 affeag1 afgfly1 afghor1 afmdov1 afpfly1 afpkin1 afpwag1 afrgos1 afrgrp1 afrjac1 afrthr1 amesun2 augbuz1 bagwea1 barswa bawhor2 bawman1 bcbeat1 beasun2 bkctch1 bkfruw1 blacra1 blacuc1 blakit1 blaplo1 blbpuf2 blcapa2 blfbus1 blhgon1 blhher1 blksaw1 blnmou1 blnwea1 bltapa1 bltbar1 bltori1 blwlap1 brcale1 brcsta1 brctch1 brcwea1 brican1 brobab1 broman1 brosun1 brrwhe3 brtcha1 brubru1 brwwar1 bswdov1 btweye2 bubwar2 butapa1 cabgre1 carcha1 carwoo1 categr ccbeat1 chespa1 chewea1 chibat1 chtapa3 chucis1 cibwar1 cohmar1 colsun2 combul2 combuz1 comsan crefra2 crheag1 crohor1 darbar1 darter3 didcuc1 dotbar1 dutdov1 easmog1 eaywag1 edcsun3 egygoo equaka1 eswdov1 eubeat1 fatrav1 fatwid1 fislov1 fotdro5 gabgos2 gargan gbesta1 gnbcam2 gnhsun1 gobbun1 gobsta5 gobwea1 golher1 grbcam1 grccra1 grecor greegr grewoo2 grwpyt1 gryapa1 grywrw1 gybfis1 gycwar3 gyhbus1 gyhkin1 gyhneg1 gyhspa1 gytbar1 hadibi1 hamerk1 hartur1 helgui hipbab1 hoopoe huncis1 hunsun2 joygre1 kerspa2 klacuc1 kvbsun1 laudov1 lawgol lesmaw1 lessts1 libeat1 litegr litswi1 litwea1 loceag1 lotcor1 lotlap1 luebus1 mabeat1 macshr1 malkin1 marsto1 marsun2 mcptit1 meypar1 moccha1 mouwag1 ndcsun2 nobfly1 norbro1 norcro1 norfis1 norpuf1 nubwoo1 pabspa1 palfly2 palpri1 piecro1 piekin1 pitwhy purgre2 pygbat1 quailf1 ratcis1 raybar1 rbsrob1 rebfir2 rebhor1 reboxp1 reccor reccuc1 reedov1 refbar2 refcro1 reftin1 refwar2 rehblu1 rehwea1 reisee2 rerswa1 rewsta1 rindov rocmar2 rostur1 ruegls1 rufcha2 sacibi2 sccsun2 scrcha1 scthon1 shesta1 sichor1 sincis1 slbgre1 slcbou1 sltnig1 sobfly1 somgre1 somtit4 soucit1 soufis1 spemou2 spepig1 spewea1 spfbar1 spfwea1 spmthr1 spwlap1 squher1 strher strsee1 stusta1 subbus1 supsta1 tacsun1 tafpri1 tamdov1 thrnig1 trobou1 varsun2 vibsta2 vilwea1 vimwea1 walsta1 wbgbir1 wbrcha2 wbswea1 wfbeat1 whbcan1 whbcou1 whbcro2 whbtit5 whbwea1 whbwhe3 whcpri2 whctur2 wheslf1 whhsaw1 whihel1 whrshr1 witswa1 wlwwar wookin1 woosan wtbeat1 yebapa1 yebbar1 yebduc1 yebere1 yebgre1 yebsto1 yeccan1 yefcan yelbis1 yenspu1 yertin1 yesbar1 yespet1 yetgre1 yewgre1".split()

class PretrainDataset2023(Dataset):
    def __init__(
            self,
            dataset_dir: str,
    ):
        ignore_labels = []  # (target_columns)
        df2021 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2021/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2022 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2022/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2023 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2023/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2021["data_year"] = 2021
        df2022["data_year"] = 2022
        df2023["data_year"] = 2023
        df = pd.concat([df2021, df2022, df2023], ignore_index=True)

        print(len(df))
        df = df[~df.primary_label.isin(ignore_labels)]
        print(len(df))
        labels = list(set(df.primary_label.unique()))
        labels.sort()
        self.labels = labels
        self.df = df
        self.dataset_dir = dataset_dir


        self.duration = 15
        self.sr = 32000
        self.dsr = self.duration * self.sr
        self.bird2id = {x: idx for idx, x in enumerate(labels)}

    def load_one(self, filename, offset, duration):
        try:
            wav, _ = librosa.load(filename, sr=None, offset=offset, duration=duration)
        except:
            print("failed reading", filename)
        return wav


    def __getitem__(self, i):
        row = self.df.iloc[i]
        data_year = row['data_year']
        filename = os.path.join(self.dataset_dir, f"birdclef-{data_year}",
                                "train_audio" if data_year != 2021 else "train_short_audio", row["primary_label"], row['filename'].split("/")[-1])

        ## wav

        wav_len_sec = librosa.get_duration(filename=filename, sr=None)
        duration = self.duration
        max_offset = wav_len_sec - duration
        max_offset = max(max_offset, 1)
        offset = np.random.randint(max_offset)


        wav = self.load_one(filename, offset=offset, duration=self.duration)
        if wav.shape[0] < (self.dsr):
            wav = np.pad(wav, (0, self.dsr - wav.shape[0]))

        ## labels
        labels = torch.zeros((len(self.labels),))
        labels[self.bird2id[row['primary_label']]] = 1.0
        for x in ast.literal_eval(row['secondary_labels']):
            try:
                labels[self.bird2id[x]] = 1.0
            except:
                continue

        return {
            "wav": torch.tensor(wav).unsqueeze(0),
            "labels": labels,
        }

    def __len__(self):
        return len(self.df)

class PretrainDatasetSED(Dataset):
    def __init__(
            self,
            dataset_dir: str,
    ):
        ignore_labels = []  # (target_columns)
        df2021 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2021/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2022 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2022/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2023 = pd.read_csv(os.path.join(dataset_dir, "birdclef-2023/train_metadata.csv"))[
            ["primary_label", "secondary_labels", "filename"]]
        df2021["data_year"] = 2021
        df2022["data_year"] = 2022
        df2023["data_year"] = 2023
        df = pd.concat([df2021, df2022, df2023], ignore_index=True)

        print(len(df))
        df = df[~df.primary_label.isin(ignore_labels)]
        print(len(df))
        labels = list(set(df.primary_label.unique()))
        labels.sort()
        self.labels = labels
        self.df = df
        self.dataset_dir = dataset_dir


        self.duration = 15
        self.sr = 32000
        self.dsr = self.duration * self.sr
        self.bird2id = {x: idx for idx, x in enumerate(labels)}

    def load_one(self, filename, offset, duration):
        try:
            wav, _ = librosa.load(filename, sr=None, offset=offset, duration=duration)
        except:
            print("failed reading", filename)
        return wav


    def __getitem__(self, i):
        row = self.df.iloc[i]
        data_year = row['data_year']
        filename = os.path.join(self.dataset_dir, f"birdclef-{data_year}",
                                "train_audio" if data_year != 2021 else "train_short_audio", row["primary_label"], row['filename'].split("/")[-1])

        ## wav

        wav_len_sec = librosa.get_duration(filename=filename, sr=None)
        duration = self.duration
        max_offset = wav_len_sec - duration
        max_offset = max(max_offset, 1)
        offset = np.random.randint(max_offset)


        wav = self.load_one(filename, offset=0, duration=None)
        if wav.shape[0] < (self.dsr):  wav = np.pad(wav, (0, self.dsr - wav.shape[0]))

        ## labels
        labels = torch.zeros((len(self.labels),))
        labels[self.bird2id[row['primary_label']]] = 1.0
        for x in ast.literal_eval(row['secondary_labels']):
            try:
                labels[self.bird2id[x]] = 0.5
            except:
                continue

        return {
            "wav": torch.tensor(wav).unsqueeze(0),
            "labels": labels,
        }

    def __len__(self):
        return len(self.df)
