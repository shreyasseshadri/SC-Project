import pandas as pd
import glob
import sys


def make_csv(path,out_csv):
    filenames=[]
    text=[]
    rating=[]
    id=[]
    ln=len(path)
    path+='/*.txt'
    print('total number of files :',len(glob.glob(path)))
    i=1
    for filename in glob.glob(path):
        filenames.append(filename)
        with open(filename,'r') as f:
            text.append(f.read().replace('\n',''))
        desc=filename[ln+1:-4].split('_')
        rating.append(desc[1])
        id.append(desc[0])
        s='file '+str(i)+' done '
        print(s,end='\r')
        i+=1
    data=pd.DataFrame({'text':text,'rating':rating,'id':id})
    data.to_csv(out_csv,index=False)
    print('csv made at ',out_csv)

if __name__ == '__main__':
    path=sys.argv[1]
    out_csv=sys.argv[2]
    make_csv(path,out_csv)