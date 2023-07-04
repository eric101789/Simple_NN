import csv
import re
from math import sqrt, atan2
from PIL import Image
import numpy as np

if __name__ == "__main__":
    """
    This script file demonstrates how to transform raw CSI out from the ESP32 into CSI-amplitude and CSI-phase.
    """
    FILE_NAME1 = "training.csv"                    # 原始 csv 檔案
    OUTPUT_FILE = "csi_amplitudes.csv"  # 輸出的 CSV 檔案名稱
    # PATH = "dataset/"

    f1 = open(FILE_NAME1)

    loop1 = 0
    loop_n = 11000                           # 設定總共要輸出多少筆資料
    amplitudes_data = []  # 儲存振幅資料

    for j1, l1 in enumerate(f1.readlines()):
        imaginary1 = []
        real1 = []
        amplitudes1 = []
        phases1 = []

        # Parse string to create integer list
        csi_string1 = re.findall(r"\[(.*)\]", l1)[0]
        csi_raw1 = [int(x) for x in csi_string1.split(" ") if x != '']

        # Create list of imaginary and real numbers from CSI
        for i in range(len(csi_raw1)):
            if i % 2 == 0:
                imaginary1.append(csi_raw1[i])
            else:
                real1.append(csi_raw1[i])

        # Transform imaginary and real into amplitude and phase
        for i in range(int(len(csi_raw1) / 2)):
            # 把計算後的振幅值取 3 位小數點後，放入 list 裡
            amplitudes1.append(format(sqrt(imaginary1[i] ** 2 + real1[i] ** 2), '.0f'))
            phases1.append(format(atan2(imaginary1[i], real1[i]), '.3f'))

        if loop1 >= loop_n:
            break

        data_sub = []
        # 將數據子載波的上、下兩部分整合成一個新的 list
        # data_sub = [element for e in [amplitudes1[6:32], amplitudes1[33:59]] for element in e]
        data_sub = amplitudes1[6:32] + amplitudes1[33:59]
        # print('數據子載波 : ', data_sub)

        amplitudes_data.append(data_sub)  # 將振幅資料儲存到列表中

        loop1 += 1

    # 將振幅資料輸出為 CSV 檔案
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in amplitudes_data:
            # writer.writerow([" ".join(row)])
            writer.writerow(row)

    print("轉換完成!")
