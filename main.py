import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import io


data_str = """frame	xmin	xmax	ymin	ymax	class_id
1478019952686311006.jpg	237	251	143	155	1
1478019952686311006.jpg	437	454	120	186	3
1478019953180167674.jpg	218	231	146	158	1
1478019953689774621.jpg	171	182	141	154	2
1478019953689774621.jpg	179	191	144	155	1
1478019953689774621.jpg	206	220	145	156	1
1478019953689774621.jpg	385	420	122	152	1
1478019953689774621.jpg	411	462	124	148	1
1478019954186238236.jpg	165	177	140	154	2
1478019954186238236.jpg	171	182	144	157	1
1478019954186238236.jpg	200	214	145	157	1
1478019954186238236.jpg	362	430	118	149	1
1478019954186238236.jpg	433	479	124	148	1
1478019954685370994.jpg	160	173	140	153	2
1478019954685370994.jpg	165	177	144	156	1
1478019954685370994.jpg	194	207	144	156	1
1478019954685370994.jpg	373	457	118	152	1
1478019955185244088.jpg	154	169	139	153	2
1478019955185244088.jpg	160	172	144	155	1
1478019955185244088.jpg	189	203	142	154	1
1478019955185244088.jpg	390	479	114	153	1
1478019955679801306.jpg	148	162	143	156	1
1478019955679801306.jpg	149	166	138	153	2
1478019955679801306.jpg	184	198	142	153	1
1478019956186247611.jpg	137	153	141	157	1
1478019956186247611.jpg	145	162	137	152	2
1478019956186247611.jpg	179	192	141	152	1
1478019956680248165.jpg	123	147	141	162	1
1478019956680248165.jpg	138	157	138	153	2
1478019956680248165.jpg	181	193	143	153	1
1478019957180061202.jpg	97	129	141	167	1
1478019957180061202.jpg	129	150	137	160	2
1478019957180061202.jpg	178	188	144	154	1
1478019957180061202.jpg	402	421	134	149	1
1478019957687018435.jpg	43	97	139	170	1
1478019957687018435.jpg	111	138	135	160	2
1478019957687018435.jpg	173	184	144	154	1
1478019957687018435.jpg	359	382	137	147	1
1478019957687018435.jpg	412	431	133	148	1
1478019957687018435.jpg	429	447	135	148	1
1478019958179775471.jpg	90	129	134	162	2
1478019958179775471.jpg	173	183	144	154	1
1478019958179775471.jpg	370	393	137	146	1
1478019958179775471.jpg	429	449	134	149	1
1478019958682197101.jpg	59	110	129	160	2
1478019958682197101.jpg	173	183	143	153	1
1478019958682197101.jpg	383	406	136	145	1
1478019959187829768.jpg	0	79	133	265	1
1478019959187829768.jpg	7	88	123	167	2
1478019959187829768.jpg	176	187	144	154	1
1478019959187829768.jpg	396	418	136	148	1
1478019959681353555.jpg	2	135	130	230	1
1478019959681353555.jpg	166	178	144	152	1
1478019959681353555.jpg	180	191	143	153	1
1478019959681353555.jpg	417	451	134	145	1
1478019960189614397.jpg	45	158	136	211	1
1478019960189614397.jpg	186	196	144	154	1
1478019960189614397.jpg	447	477	135	145	1
1478019960680764792.jpg	78	168	137	196	1
1478019960680764792.jpg	185	196	141	151	1
1478019961182003465.jpg	112	177	139	182	1
1478019961680640592.jpg	116	138	146	158	1
1478019961680640592.jpg	129	186	139	179	1
1478019961680640592.jpg	264	271	139	155	3
1478019961680640592.jpg	366	386	135	145	1
1478019961680640592.jpg	423	438	134	146	1
1478019962181150666.jpg	116	140	146	157	1
1478019962181150666.jpg	133	153	141	159	1
1478019962181150666.jpg	147	194	140	176	1
1478019962181150666.jpg	268	275	138	156	3
1478019962181150666.jpg	377	397	134	144	1
1478019962181150666.jpg	402	425	134	145	1
1478019962181150666.jpg	453	457	132	144	1
1478019962681840550.jpg	95	126	141	161	1
1478019962681840550.jpg	116	141	146	157	1
1478019962681840550.jpg	157	198	140	172	1
1478019962681840550.jpg	271	280	137	156	3
1478019962681840550.jpg	384	394	134	143	1
1478019962681840550.jpg	417	439	134	145	1
1478019962681840550.jpg	440	465	132	144	1
1478019963181283434.jpg	117	143	146	157	1
1478019963181283434.jpg	167	202	141	168	1
1478019963181283434.jpg	278	287	138	158	3
1478019963181283434.jpg	399	416	134	143	1
1478019963181283434.jpg	437	460	132	143	1
1478019963682173845.jpg	86	113	146	156	1
1478019963682173845.jpg	117	143	146	157	1
1478019963682173845.jpg	173	206	140	165	1
1478019963682173845.jpg	285	295	138	160	3
1478019963682173845.jpg	407	434	135	144	1
1478019964181479375.jpg	52	79	142	156	1
1478019964181479375.jpg	113	138	142	155	1
1478019964181479375.jpg	180	209	139	162	1
1478019964181479375.jpg	295	306	133	155	3
1478019964687995430.jpg	100	125	144	163	1
1478019964687995430.jpg	184	210	139	162	1
1478019964687995430.jpg	306	319	137	160	3
1478019965181415731.jpg	75	109	141	163	1
1478019965181415731.jpg	189	212	139	162	1
1478019965181415731.jpg	219	229	142	150	1
1478019965181415731.jpg	230	235	127	135	5
1478019965181415731.jpg	255	271	141	151	1
1478019965181415731.jpg	328	342	132	164	3
1478019965682301515.jpg	188	215	139	161	1
1478019965682301515.jpg	220	230	143	151	1
1478019965682301515.jpg	231	237	127	135	5
1478019965682301515.jpg	258	274	141	151	1
1478019965682301515.jpg	350	367	127	168	3
1478019966187511711.jpg	188	218	139	161	1
1478019966187511711.jpg	220	230	142	150	1
1478019966187511711.jpg	231	237	126	135	5
1478019966187511711.jpg	259	275	141	151	1
1478019966688931929.jpg	188	197	141	153	1
1478019966688931929.jpg	193	218	141	159	1
1478019966688931929.jpg	219	229	142	150	1
1478019966688931929.jpg	233	238	126	134	5
1478019967180239928.jpg	180	195	143	154	1
1478019967180239928.jpg	190	215	141	159	1
1478019967180239928.jpg	219	228	142	150	1
1478019967180239928.jpg	230	236	125	134	5
1478019967686224547.jpg	167	185	143	156	1
1478019967686224547.jpg	188	212	142	160	1
1478019967686224547.jpg	214	225	142	152	1
1478019967686224547.jpg	227	232	126	134	5
1478019968180276750.jpg	153	174	142	158	1
1478019968180276750.jpg	186	211	142	160	1
1478019968180276750.jpg	212	222	144	153	1
1478019968180276750.jpg	225	230	126	134	5
1478019968680240537.jpg	134	160	140	159	1
1478019968680240537.jpg	184	204	142	160	1
1478019968680240537.jpg	207	219	142	151	1
1478019968680240537.jpg	208	220	144	151	1
1478019968680240537.jpg	222	227	124	133	5
1478019969186707568.jpg	101	135	140	162	1
1478019969186707568.jpg	182	202	143	161	1
1478019969186707568.jpg	204	215	144	152	1
1478019969186707568.jpg	204	217	143	153	1
1478019969186707568.jpg	218	224	124	132	5
1478019969688638443.jpg	46	95	138	169	1
1478019969688638443.jpg	177	198	142	159	1
1478019969688638443.jpg	198	212	142	153	1
1478019969688638443.jpg	199	211	145	152	1
1478019969688638443.jpg	214	220	123	131	5
1478019970188563338.jpg	153	158	131	140	5
1478019970188563338.jpg	174	195	143	160	1
1478019970188563338.jpg	194	207	142	153	1
1478019970188563338.jpg	195	206	144	151	1
1478019970188563338.jpg	210	216	121	130	5
1478019970680532186.jpg	146	151	132	140	5
1478019970680532186.jpg	170	191	142	159	1
1478019970680532186.jpg	189	203	141	152	1
1478019970680532186.jpg	190	202	143	150	1
1478019970680532186.jpg	206	212	119	129	5
1478019971185917857.jpg	141	146	130	138	5
1478019971185917857.jpg	169	190	141	158	1
1478019971185917857.jpg	188	202	140	151	1
1478019971185917857.jpg	189	200	141	149	1
1478019971185917857.jpg	205	211	117	126	5
1478019971686116476.jpg	136	142	129	137	5
1478019971686116476.jpg	146	159	142	151	4
1478019971686116476.jpg	166	187	140	157	1
1478019971686116476.jpg	185	189	116	125	5
1478019971686116476.jpg	186	200	139	150	1
1478019971686116476.jpg	188	199	140	148	1
1478019971686116476.jpg	204	209	115	123	5
1478019971686116476.jpg	204	210	114	123	5
1478019971686116476.jpg	214	227	135	150	4
1478019971686116476.jpg	241	249	119	129	5
1478019971686116476.jpg	265	306	135	149	1
1478019972180014279.jpg	124	138	142	152	4
1478019972180014279.jpg	132	137	128	136	5
1478019972180014279.jpg	163	184	140	157	1
1478019972180014279.jpg	182	186	115	124	5
1478019972180014279.jpg	184	197	140	151	1
1478019972180014279.jpg	185	196	139	147	1
1478019972180014279.jpg	202	207	113	121	5
1478019972180014279.jpg	203	209	112	121	5
1478019972180014279.jpg	242	250	118	128	5
1478019972180014279.jpg	268	300	133	149	1
1478019972180014279.jpg	275	283	137	160	3
1478019972685986697.jpg	34	65	144	157	1
1478019972685986697.jpg	104	117	141	152	4
1478019972685986697.jpg	125	130	126	135	5
1478019972685986697.jpg	162	184	139	157	1
1478019972685986697.jpg	179	183	114	123	5
1478019972685986697.jpg	180	197	140	150	1
1478019972685986697.jpg	182	193	139	146	1
1478019972685986697.jpg	201	206	111	119	5
1478019972685986697.jpg	201	207	110	120	5
1478019972685986697.jpg	242	250	115	128	5
1478019972685986697.jpg	270	302	131	148	1
1478019972685986697.jpg	275	285	135	154	3
1478019973185520968.jpg	48	79	143	156	1
1478019973185520968.jpg	50	77	143	154	1
1478019973185520968.jpg	83	97	141	153	4
1478019973185520968.jpg	118	125	125	134	5
1478019973185520968.jpg	157	184	138	155	1
1478019973185520968.jpg	175	180	111	120	5
1478019973185520968.jpg	178	196	139	151	1"""


data = pd.read_csv(io.StringIO(data_str), sep='\t')


def perform_eda(df):
    print("="*50)
    print("TEMEL BİLGİLER")
    print("="*50)
    print(f"Toplam kayıt sayısı: {len(df)}")
    print(f"Benzersiz görüntü sayısı: {df['frame'].nunique()}")
    print(f"Sınıf sayısı: {df['class_id'].nunique()}")
    print("\nİlk 5 kayıt:")
    print(df.head())
    
    
    print("\n" + "="*50)
    print("SINIF DAĞILIMI")
    print("="*50)
    class_dist = df['class_id'].value_counts().sort_index()
    print(class_dist)
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='class_id', data=df)
    plt.title('Sınıf Dağılımı')
    plt.xlabel('Sınıf ID')
    plt.ylabel('Nesne Sayısı')
    plt.show()
    
    
    df['width'] = df['xmax'] - df['xmin']
    df['height'] = df['ymax'] - df['ymin']
    df['area'] = df['width'] * df['height']
    df['aspect_ratio'] = df['width'] / df['height']
    
    print("\n" + "="*50)
    print("BOUNDING BOX İSTATİSTİKLERİ")
    print("="*50)
    print(df[['width', 'height', 'area', 'aspect_ratio']].describe())
    
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(x='class_id', y='width', data=df)
    plt.title('Genişlik Dağılımı')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(x='class_id', y='height', data=df)
    plt.title('Yükseklik Dağılımı')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(x='class_id', y='area', data=df)
    plt.title('Alan Dağılımı')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(x='class_id', y='aspect_ratio', data=df)
    plt.title('En-Boy Oranı Dağılımı')
    
    plt.tight_layout()
    plt.show()
    
    
    objects_per_image = df['frame'].value_counts()
    print("\n" + "="*50)
    print("GÖRÜNTÜ BAŞINA NESNE SAYISI")
    print("="*50)
    print(objects_per_image.describe())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(objects_per_image, bins=20)
    plt.title('Görüntü Başına Nesne Sayısı Dağılımı')
    plt.xlabel('Nesne Sayısı')
    plt.ylabel('Görüntü Sayısı')
    plt.show()
    
    
    df['x_center'] = (df['xmin'] + df['xmax']) / 2
    df['y_center'] = (df['ymin'] + df['ymax']) / 2
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='x_center', y='y_center', hue='class_id', data=df, alpha=0.6)
    plt.title('Nesne Merkezlerinin Dağılımı')
    plt.xlabel('X Koordinatı')
    plt.ylabel('Y Koordinatı')
    plt.show()


perform_eda(data)

pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None) 
print(data)


import pandas as pd # type: ignore
from io import StringIO


data = """frame	xmin	xmax	ymin	ymax	class_id
img1.jpg	100	300	150	350	1
img1.jpg	200	400	250	450	2
img2.jpg	150	350	200	400	1"""


df = pd.read_csv(StringIO(data), sep='\t')


assert not df.empty, "Veri çerçevesi boş!"
assert 'class_id' in df.columns, "class_id sütunu eksik!"

print("Veri başarıyla yüklendi:")
print(df.head())


import pandas as pd # type: ignore


print(" Veri Boyutu:", df.shape)
print("Sütunlar:", df.columns.tolist())


print("\n Eksik Değerler:")
print(df.isnull().sum())


print("\n Sınıf Dağılımı:")
print(df['class_id'].value_counts())


df = df[(df['xmax'] > df['xmin']) & (df['ymax'] > df['ymin'])]


df['width'] = df['xmax'] - df['xmin']
df['height'] = df['ymax'] - df['ymin']
df['area'] = df['width'] * df['height']

print("\n Bounding Box İstatistikleri:")
print(df[['width', 'height', 'area']].describe())


img_width, img_height = 416, 416  

df['x_center'] = ((df['xmin'] + df['xmax']) / 2) / img_width
df['y_center'] = ((df['ymin'] + df['ymax']) / 2) / img_height
df['norm_width'] = df['width'] / img_width
df['norm_height'] = df['height'] / img_height

print("\n YOLO Formatı Örneği:")
print(df[['frame', 'class_id', 'x_center', 'y_center', 'norm_width', 'norm_height']].head())

from sklearn.model_selection import train_test_split # type: ignore

train, test = train_test_split(df['frame'].unique(), test_size=0.2, random_state=42)

train_df = df[df['frame'].isin(train)]
test_df = df[df['frame'].isin(test)]

print("\n Bölünmüş Veri Boyutları:")
print(f"Eğitim: {len(train_df)} kayıt")
print(f"Test: {len(test_df)} kayıt")

import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

unique_images = df['frame'].unique()
print(f"Toplam benzersiz görüntü sayısı: {len(unique_images)}")

train_images, test_images = train_test_split(unique_images, test_size=0.2, random_state=42)

train_df = df[df['frame'].isin(train_images)]
test_df = df[df['frame'].isin(test_images)]

print(f"\n Bölünmüş Veri Boyutları:")
print(f"Toplam kayıt: {len(df)}")
print(f"Eğitim görüntüleri: {len(train_images)} ({len(train_df)} kayıt)")
print(f"Test görüntüleri: {len(test_images)} ({len(test_df)} kayıt)")


print("\n Sınıf Dağılımı (Eğitim):")
print(train_df['class_id'].value_counts())
print("\n Sınıf Dağılımı (Test):")
print(test_df['class_id'].value_counts())



import pandas as pd # type: ignore


data = {
    "xmin": [237, 437, 218, 171, 179, 206, 385, 411, 165, 171],
    "xmax": [251, 454, 231, 182, 191, 220, 420, 462, 177, 182],
    "ymin": [143, 120, 146, 141, 144, 145, 122, 124, 140, 144],
    "ymax": [155, 186, 158, 154, 155, 156, 152, 148, 154, 157],
    "class_id": [1, 3, 1, 2, 1, 1, 1, 1, 2, 1]
}

df = pd.DataFrame(data)


df.insert(0, "frame", range(1, len(df) + 1))

print("📌 Sütunlar:", df.columns.tolist())
print("📌 Örnek veri:\n", df.head())


required_cols = ['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']
missing_cols = [col for col in required_cols if col not in df.columns]
assert not missing_cols, f"Eksik sütunlar var: {missing_cols}"


from ultralytics import YOLO # type: ignore


model = YOLO('yolov5n.pt')  


model.train(
    data="C:/Users/Sefa/Downloads/roboflow_dataset",  
    epochs=50,        
    imgsz=640,       
    batch=16,         
    project="weights", 
    name="yolov5_custom" 
)


import torch # type: ignore
from yolov5.train import run # type: ignore

PROJECT_PATH = r"C:\yolo_proje" 

run(
    data=fr"{PROJECT_PATH}\data.yaml",
    weights='yolov5s.pt',
    epochs=50,  
    batch_size=16,
    imgsz=640,  
    name='car_detection',
    device='0' if torch.cuda.is_available() else 'cpu'  
)


import torch # type: ignore
from yolov5.train import run # type: ignore


run(
    data='C:/Users/Sefa Üğücü/Downloads/My First Project.v1-yolov5-pytorch.yolov8/data.yaml',
    weights='yolov5s.pt',  
    epochs=50,
    batch_size=16,
    imgsz=640,
    name='car_detection',
    device='0' if torch.cuda.is_available() else 'cpu'
)
