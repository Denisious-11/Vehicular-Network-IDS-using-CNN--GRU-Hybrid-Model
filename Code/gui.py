#Iporting libraries

import time
import joblib
import pandas as pd
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from customtkinter import *
from keras.models import model_from_json

import warnings
warnings.filterwarnings('ignore')

# Creating window
root = Tk()
root.geometry('1300x700')
root.resizable(0,0)
root.title('Vehicular Network')
root.configure(bg='#bcebf5')

df = pd.read_csv('../Dataset/test_set.csv')        # Read test dataset


json_path = '../SavedFiles/arch.json'

# Load the JSON file
with open(json_path, 'r') as json_file:
    loaded_model_json = json_file.read()

# Reconstruct the model architecture from JSON
model = model_from_json(loaded_model_json)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('../SavedFiles/model.h5')     # Load model

# mfile = model9
# mfile.load_weights('/content/MyDrive/Vehicle/model9.h5')
param = joblib.load('../SavedFiles/features.pkl')      # Load feature selection
scale = joblib.load('../SavedFiles/scale.pkl')        # Load standard scaler


fload = Frame(root,width=1300,height=700,bg='#bcebf5')
fresult = Frame(root,width=1300,height=700,bg='#bcebf5')
nfresult = Frame(root,width=1300,height=700,bg='#bcebf5')


set_default_color_theme("green")

viz= Frame(root,width=1300,height=700,bg='#bcebf5')

def loading():      # progress bar for analyzing
    fload.pack(expand=True,fill=BOTH)
    inter.insert(END,'\nAnalyzing inputs....')

    progress_label = Label(fload,text='Analyzing .....',font=('Trobuchet',12,'bold'),bg='#bcebf5')
    progress_label.place(x=330,y=375)

    progress = ttk.Style()
    progress.theme_use('clam')
    progress.configure('red.Horizontal.TProgressbar',background='#eb4034',thickness=35)
    
    load = ttk.Progressbar(fload,orient=HORIZONTAL,mode='determinate',length=300,style='red.Horizontal.TProgressbar')
    load.place(x=270,y=400)
    time.sleep(1)
    load['value']=20
    fload.update_idletasks()
    time.sleep(1)
    load['value']=40
    fload.update_idletasks()
    time.sleep(1)
    load['value']=60
    fload.update_idletasks()
    time.sleep(1)
    load['value']=80
    fload.update_idletasks()
    time.sleep(1)
    load['value']=100

def input():      # Read data for Prediction

    
    start_time = time.time()

    selected = tree1.focus()
    vals = tree1.item(selected,'values')
    ts = vals[0]
    print(ts)
    proto = vals[1]
    duration = vals[2]
    src = vals[3]
    dst = vals[4]
    conn = vals[5]
    missed = vals[6]
    src_pkt = vals[7]
    src_ip = vals[8]
    dst_pkt = vals[9]
    dst_ip = vals[10]
    qclass = vals[11]
    qtype = vals[12]
    rcode = vals[13]
    inter.insert(END,f'The value of ts is {ts} ')
    inter.insert(END,f'\nThe value of proto is {proto} ')
    inter.insert(END,f'\nThe value of duration is {duration} ')
    inter.insert(END,f'\nThe value of src_bytes is {src} ')
    inter.insert(END,f'\nThe value of dst_bytes is {dst} ')
    inter.insert(END,f'\nThe value of conn_state is {conn} ')
    inter.insert(END,f'\nThe value of missing_bytes is {missed} ')
    inter.insert(END,f'\nThe value of src_pkts is {src_pkt} ')
    inter.insert(END,f'\nThe value of src_ip_bytes is {src_ip} ')
    inter.insert(END,f'\nThe value of dst_pkts is {dst_pkt} ')
    inter.insert(END,f'\nThe value of dst_ip_bytes is {dst_ip}')
    inter.insert(END,f'\nThe value of dns_qclass is {qclass}' )
    inter.insert(END,f'\nThe value of dns_qtype is {qtype} ')
    inter.insert(END,f'\nThe value of dns_rcode is {rcode}')

    loading()
    fload.pack_forget()

    rowDf = pd.DataFrame([pd.Series([ts, proto, duration, src, dst, conn, missed, src_pkt, src_ip, dst_pkt, dst_ip, qclass, qtype, rcode])])
    print(rowDf)
    inter.insert(END,f'\n----------------------------------------\nConverted into dataframe: {rowDf}')
    # new_rowDf = pd.DataFrame(sc.fit_transform(rowDF))
    # print('scale',new_rowDf)
    
    inter.insert(END,f'\nFeature selecting...')
    new_rowDf = pd.DataFrame(param.transform(rowDf))
    print(f'{new_rowDf}')
    inter.insert(END,f'\nAfter feature selection: {new_rowDf}')


    inter.insert(END,f'\nFeature scaling...')
    new_rowDf = scale.transform(new_rowDf)
    print('scale: ',new_rowDf)
    inter.insert(END,f'\nAfter feature scaling: {new_rowDf}')


    prediction = model.predict(new_rowDf)
    print(prediction)
    inter.insert(END,f'\n----------------------------------------\n')
    inter.insert(END,f'\nPrediction: {prediction}')
    
    print(f'Prediction values are {prediction[0][0]}')
    
    if prediction > 0.5:
        percentage = prediction[0][0]*100
        print(percentage,'Attack')

        nfresult.pack_forget()
        fresult.pack(expand=True,fill=BOTH)
        result_attack.config(text='ATTACK')

        inter.insert(END,f'\nAttack : {percentage}')
    else:
        percentage = prediction[0][0]*100
        print(percentage,'Normal')

        fresult.pack_forget()
        nfresult.pack(expand=True,fill=BOTH)
        result_normal.config(text='NORMAL')

        inter.insert(END,f'\nNormal : {percentage}')
        

    elapsed_time = time.time() - start_time
    inter.insert(END,f'\nTotal time taken: {elapsed_time}secs')

def predict():  # Prediction 
    inter.delete(1.0,END)
    fresult.pack_forget()
    nfresult.pack_forget()
    fload.pack_forget()
    selected_item = tree1.focus() 
    input_text = inp.get("1.0", "end-1c")
    if selected_item and input_text:
        input()  # Perform prediction if a row is selected
    else:
        messagebox.showwarning('Input','Please select a row from the data')
    # input()

def clearData():        # Clear selection, texts, and result
    
    inter.delete(1.0,END)  
    inp.delete(1.0,END)
    tree1.selection_remove(tree1.selection())

    fresult.pack_forget()
    nfresult.pack_forget()
  
#     tree1.see(tree1.get_children()[0]
   

def select_row(event):      # Select data from treeview
    inp.delete(1.0,END)
    selected = tree1.focus()
    vals = tree1.item(selected,'values')

    inp.insert(END,f'ts : {vals[0]}')
    inp.insert(END,f'\nproto : {vals[1]}')
    inp.insert(END,f'\nduration : {vals[2]}')
    inp.insert(END,f'\nsrc_bytes : {vals[3]}')
    inp.insert(END,f'\ndst_bytes : {vals[4]}')
    inp.insert(END,f'\nconn_state : {vals[5]}')
    inp.insert(END,f'\nmissing_bytes : {vals[6]}')
    inp.insert(END,f'\nsrc_pkts : {vals[7]}')
    inp.insert(END,f'\nsrc_ip_bytes : {vals[8]}')
    inp.insert(END,f'\ndst_pkts : {vals[9]}')
    inp.insert(END,f'\ndst_ip_bytes : {vals[10]}')
    inp.insert(END,f'\ndns_qclass : {vals[11]}')
    inp.insert(END,f'\ndns_qtype : {vals[12]}')
    inp.insert(END,f'\ndnc_rcode : {vals[13]}')

  

Label(root,bg='#bcebf5',text='Vehicle Intrusion Detection',font=('Trobuchet',18,'bold'),fg='#cf5611').place(x=600,y=20)


Label(root,bg='#bcebf5',text='Input').place(x=50,y=160)
inp = CTkTextbox(root)
inp.place(x=120,y=70)


Label(root,text='Result',font=(('Trobuchet',13,'bold')),bg='#bcebf5').place(x=200,y=400)


result_attack = Label(fresult,bg='#bcebf5',text='',font=(('Helvetica',13,'bold')),fg='red')
result_attack.place(x=330,y=400)

result_normal = Label(nfresult,bg='#bcebf5',text='',font=(('Helvetica',13,'bold')),fg='red')
result_normal.place(x=330,y=400)

Label(root,bg='#bcebf5',text='Process').place(x=800,y=360)
inter = Text(root,width=40,height=10)       # Text to display processes
inter.place(x=800,y=380)

cols_to_show = ['ts','proto','duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes','dns_qclass','dns_qtype','dns_rcode']
def data(treeview, dataframe,columns):      # Display data on treeview
    treeview.delete(*treeview.get_children())
    for row in dataframe[columns].itertuples(index=False):
        treeview.insert('',END,values=row)


cols_1 = ('ts','proto','duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts','dst_ip_bytes','dns_qclass','dns_qtype','dns_rcode') 
tree1 = ttk.Treeview(root,columns=cols_1,show='headings')       # Treeview
tree1.heading('ts',anchor=W,text='ts')
tree1.column('ts',stretch=NO,minwidth=0,width=70)
tree1.heading('proto',anchor=W,text='proto')
tree1.column('proto',stretch=NO,minwidth=0,width=40)
tree1.heading('duration',anchor=W,text='duration')
tree1.column('duration',stretch=NO,minwidth=0,width=70)
tree1.heading('src_bytes',anchor=W,text='src_bytes')
tree1.column('src_bytes',stretch=NO,minwidth=0,width=60)
tree1.heading('dst_bytes',anchor=W,text='dst_bytes')
tree1.column('dst_bytes',stretch=NO,minwidth=0,width=60)
tree1.heading('conn_state',anchor=W,text='conn_state')
tree1.column('conn_state',stretch=NO,minwidth=0,width=60)
tree1.heading('missed_bytes',anchor=W,text='missed_bytes')
tree1.column('missed_bytes',stretch=NO,minwidth=0,width=60)
tree1.heading('src_pkts',anchor=W,text='src_pkts')
tree1.column('src_pkts',stretch=NO,minwidth=0,width=60)
tree1.heading('src_ip_bytes',anchor=W,text='src_ip_bytes')
tree1.column('src_ip_bytes',stretch=NO,minwidth=0,width=60)
tree1.heading('dst_pkts',anchor=W,text='dst_pkts')
tree1.column('dst_pkts',stretch=NO,minwidth=0,width=60)
tree1.heading('dst_ip_bytes',anchor=W,text='dst_ip_bytes')
tree1.column('dst_ip_bytes',stretch=NO,minwidth=0,width=60)
tree1.heading('dns_qclass',anchor=W,text='dns_qclass')
tree1.column('dns_qclass',stretch=NO,minwidth=0,width=60)
tree1.heading('dns_qtype',anchor=W,text='dns_qtype')
tree1.column('dns_qtype',stretch=NO,minwidth=0,width=60)
tree1.heading('dns_rcode',anchor=W,text='dns_rcode')
tree1.column('dns_rcode',stretch=NO,minwidth=0,width=60)


data(tree1,df,cols_to_show)     # Calling data to display
si_1 = 1


scrolly1 = Scrollbar(root,orient=VERTICAL)
scrolly1.place(x=1190,y=70,width=20,height=230)
scrolly1.configure(command=tree1.yview)
tree1.configure(yscrollcommand=scrolly1.set)

tree1.place(x=350,y=70)

tree1.bind('<ButtonRelease-1>',select_row)

submit = CTkButton(root,text='Predict',width=10,command=predict)        # Button to predict
submit.place(x=300,y=600)

clear = CTkButton(root,text='Clear',width=10,command=clearData)     # Button to clear data
clear.place(x=400,y=600)




root.mainloop()     # End of the window 