import pandas as pd
import numpy as np


xl_file = pd.ExcelFile('./UN_MigrantStockTotal_2015.xlsx')
#--------------------------------------------------------------------------------
# Extracting Table: Contents

Content = xl_file.parse('CONTENTS')
Content = Content.tail(8)
Content = pd.DataFrame(Content).reset_index(drop = True)
Content.rename(columns = {'Unnamed: 0':'Variable',
                          'Unnamed: 1':'Desc'
                         }, inplace = True)


#--------------------------------------------------------------------------------
# Extracting Table: ANNEX

Annex = xl_file.parse('ANNEX')
Annex = Annex.tail(232)
Annex = pd.DataFrame(Annex).reset_index(drop = True)

#ANNEX.head(300)
Annex.rename(columns = {'Unnamed: 0':'Country_Code',
                        'Unnamed: 1':'Country_or_Area',
                        'Unnamed: 2':'C_Sort_Order',
                        'Unnamed: 3':'Major_Area',
                        'Unnamed: 4':'M_Code',
                        'Unnamed: 5':'M_Sort_Order',
                        'Unnamed: 6':'Region',
                        'Unnamed: 7':'R_Code',
                        'Unnamed: 8':'R_Sort_Order',
                        'Unnamed: 9':'Developed_Region',
                        'Unnamed: 10':'Least_Dev_Country',
                        'Unnamed: 11':'Sub_Saharan_Africa'
                         }, inplace = True)

Sub_Annex = Annex[['Country_Code','Developed_Region', 'Least_Dev_Country', 'Sub_Saharan_Africa']]

#--------------------------------------------------------------------------------
# Extracting Table: Notes


xl_file = pd.ExcelFile('./UN_MigrantStockTotal_2015.xlsx')
Notes = xl_file.parse('NOTES')
Notes = Notes.tail(27)
Notes = pd.DataFrame(Notes).reset_index(drop = True)

Notes.rename(columns = {'Unnamed: 0':'Note',
                        'Unnamed: 1':'Note_Desc',
                         }, inplace = True)

#------------------------------------------------------------------------------------
#Extracting Table: Table 1

Table_1 = xl_file.parse('Table 1')
Table_1 = Table_1.tail(265)
Table_1.head(265)


Table_1.rename(columns = {'Unnamed: 0':'C_Sort_Order',
                          'Unnamed: 1' :'Country_or_Area',
                          'Unnamed: 2' :'Note',
                          'Unnamed: 3' :'Country_Code',
                          'Unnamed: 4' :'Type_of_Data',
                          'Unnamed: 5' :'T1_T_1990',
                          'Unnamed: 6' :'T1_T_1995',
                          'Unnamed: 7' :'T1_T_2000',
                          'Unnamed: 8' :'T1_T_2005',
                          'Unnamed: 9' :'T1_T_2010',
                          'Unnamed: 10':'T1_T_2015',
                          'Unnamed: 11':'T1_M_1990',
                          'Unnamed: 12':'T1_M_1995',
                          'Unnamed: 13':'T1_M_2000',
                          'Unnamed: 14':'T1_M_2005',
                          'Unnamed: 15':'T1_M_2010',
                          'Unnamed: 16':'T1_M_2015',
                          'Unnamed: 17':'T1_F_1990',
                          'Unnamed: 18':'T1_F_1995',
                          'Unnamed: 19':'T1_F_2000',
                          'Unnamed: 20':'T1_F_2005',
                          'Unnamed: 21':'T1_F_2010',
                          'Unnamed: 22':'T1_F_2015',                         
                         }, inplace = True)

#Table_1.to_csv('Table_1.csv', index = False)

#------------------------------------------------------------------------------------
#Extracting Table: Table 2

Table_2 = xl_file.parse('Table 2')
Table_2 = Table_2.tail(265)
Table_2.head(265)


Table_2.rename(columns = {'Unnamed: 0':'C_Sort_Order',
                          'Unnamed: 1' :'Country_or_Area',
                          'Unnamed: 2' :'Note',
                          'Unnamed: 3' :'Country_Code',
                          'Unnamed: 4' :'T2_T_1990',
                          'Unnamed: 5' :'T2_T_1995',
                          'Unnamed: 6' :'T2_T_2000',
                          'Unnamed: 7' :'T2_T_2005',
                          'Unnamed: 8' :'T2_T_2010',
                          'Unnamed: 9' :'T2_T_2015',
                          'Unnamed: 10':'T2_M_1990',
                          'Unnamed: 11':'T2_M_1995',
                          'Unnamed: 12':'T2_M_2000',
                          'Unnamed: 13':'T2_M_2005',
                          'Unnamed: 14':'T2_M_2010',
                          'Unnamed: 15':'T2_M_2015',
                          'Unnamed: 16':'T2_F_1990',
                          'Unnamed: 17':'T2_F_1995',
                          'Unnamed: 18':'T2_F_2000',
                          'Unnamed: 19':'T2_F_2005',
                          'Unnamed: 20':'T2_F_2010',
                          'Unnamed: 21':'T2_F_2015',                         
                         }, inplace = True)

#Table_2.to_csv('Table_2.csv', index = False)


#------------------------------------------------------------------------------------
#Extracting Table: Table 3

Table_3 = xl_file.parse('Table 3')
Table_3 = Table_3.tail(265)
Table_3.head(265)


Table_3.rename(columns = {'Unnamed: 0':'C_Sort_Order',
                          'Unnamed: 1' :'Country_or_Area',
                          'Unnamed: 2' :'Note',
                          'Unnamed: 3' :'Country_Code',
                          'Unnamed: 4' :'Type_of_Data',
                          'Unnamed: 5' :'T3_T_1990',
                          'Unnamed: 6' :'T3_T_1995',
                          'Unnamed: 7' :'T3_T_2000',
                          'Unnamed: 8' :'T3_T_2005',
                          'Unnamed: 9' :'T3_T_2010',
                          'Unnamed: 10':'T3_T_2015',
                          'Unnamed: 11':'T3_M_1990',
                          'Unnamed: 12':'T3_M_1995',
                          'Unnamed: 13':'T3_M_2000',
                          'Unnamed: 14':'T3_M_2005',
                          'Unnamed: 15':'T3_M_2010',
                          'Unnamed: 16':'T3_M_2015',
                          'Unnamed: 17':'T3_F_1990',
                          'Unnamed: 18':'T3_F_1995',
                          'Unnamed: 19':'T3_F_2000',
                          'Unnamed: 20':'T3_F_2005',
                          'Unnamed: 21':'T3_F_2010',
                          'Unnamed: 22':'T3_F_2015',                         
                         }, inplace = True)

#Table_3.to_csv('Table_3.csv', index = False)


#------------------------------------------------------------------------------------
#Extracting Table: Table 4

Table_4 = xl_file.parse('Table 4')
Table_4 = Table_4.tail(265)

Table_4.rename(columns = {'Unnamed: 0':'C_Sort_Order',
                          'Unnamed: 1' :'Country_or_Area',
                          'Unnamed: 2' :'Note',
                          'Unnamed: 3' :'Country_Code',
                          'Unnamed: 4' :'Type_of_Data',
                          'Unnamed: 5' :'T4_F_1990',
                          'Unnamed: 6' :'T4_F_1995',
                          'Unnamed: 7' :'T4_F_2000',
                          'Unnamed: 8' :'T4_F_2005',
                          'Unnamed: 9' :'T4_F_2010',
                          'Unnamed: 10':'T4_F_2015',
                         }, inplace = True)

#Table_4.to_csv('Table_4.csv', index = False)
Table_4.head()

#------------------------------------------------------------------------------------
#Extracting Table: Table 5

Table_5 = xl_file.parse('Table 5')
Table_5 = Table_5.tail(265)

Table_5.rename(columns = {'Unnamed: 0':'C_Sort_Order',
                          'Unnamed: 1' :'Country_or_Area',
                          'Unnamed: 2' :'Note',
                          'Unnamed: 3' :'Country_Code',
                          'Unnamed: 4' :'Type_of_Data',
                          'Unnamed: 5' :'T5_T_199095',
                          'Unnamed: 6' :'T5_T_199500',
                          'Unnamed: 7' :'T5_T_200005',
                          'Unnamed: 8' :'T5_T_200510',
                          'Unnamed: 9' :'T5_T_201015',
                          'Unnamed: 10':'T5_M_199095',
                          'Unnamed: 11':'T5_M_199500',
                          'Unnamed: 12':'T5_M_200005',
                          'Unnamed: 13':'T5_M_200510',
                          'Unnamed: 14':'T5_M_201015',
                          'Unnamed: 15':'T5_F_199095',
                          'Unnamed: 16':'T5_F_199500',
                          'Unnamed: 17':'T5_F_200005',
                          'Unnamed: 18':'T5_F_200510',
                          'Unnamed: 19':'T5_F_201015',
                         }, inplace = True)

#Table_5.to_csv('Table_5.csv', index = False)


#------------------------------------------------------------------------------------
#Extracting Table: Table 6

Table_6 = xl_file.parse('Table 6')
Table_6 = Table_6.tail(265)

Table_6.rename(columns = {'Unnamed: 0':'C_Sort_Order',
                          'Unnamed: 1' :'Country_or_Area',
                          'Unnamed: 2' :'Note',
                          'Unnamed: 3' :'Country_Code',
                          'Unnamed: 4' :'Type_of_Data',
                          'Unnamed: 5' :'T6_TT_199095',
                          'Unnamed: 6' :'T6_TT_199500',
                          'Unnamed: 7' :'T6_TT_200005',
                          'Unnamed: 9' :'T6_TT_200510',
                          'Unnamed: 10':'T6_TT_201015',
                          'Unnamed: 11':'T6_PR_199095',
                          'Unnamed: 12':'T6_PR_199500',
                          'Unnamed: 13':'T6_PR_200005',
                          'Unnamed: 14':'T6_PR_200510',
                          'Unnamed: 15':'T6_PR_201015',
                          'Unnamed: 17':'T6_RC_199095',
                          'Unnamed: 18':'T6_RC_199500',
                          'Unnamed: 19':'T6_RC_200005',
                          'Unnamed: 20':'T6_RC_200510',
                          'Unnamed: 21':'T6_RC_201015',

                         }, inplace = True)

#Table_6.to_csv('Table_6.csv', index = False)


#----------------------------------------------------------------------------------
# create a new Column: Group to categorize Country_or_Area in Table_1

for i in Table_1['Country_or_Area']:
    for j in Annex['Country_or_Area']:
        if i == j: 
            Table_1.loc[Table_1['Country_or_Area'] == i,'Group'] = 'Country_or_Area'

for i in Table_1['Country_or_Area']:
    for j in Annex['Major_Area']:
        if i == j: 
            Table_1.loc[Table_1['Country_or_Area'] == i,'Group'] = 'Major_Area'            

for i in Table_1['Country_or_Area']:
    for j in Annex['Region']:
        if i == j: 
            Table_1.loc[Table_1['Country_or_Area'] == i,'Group'] = 'Region'               

            
Table_1.loc[Table_1.Country_or_Area == 'WORLD','Group'] = 'WORLD'
Table_1.loc[Table_1.Country_or_Area == 'Developed regions','Group'] = 'Developed regions'
Table_1.loc[Table_1.Country_or_Area == 'Developing regions','Group'] = 'Developing regions'
Table_1.loc[Table_1.Country_or_Area == 'Least developed countries','Group'] = 'Least developed countries'
Table_1.loc[Table_1.Country_or_Area == 'Less developed regions excluding least developed countries','Group'] = 'Less developed regions'
Table_1.loc[Table_1.Country_or_Area == 'Sub-Saharan Africa','Group'] = 'Sub-Saharan Africa'


Annex_2 = Table_1[['C_Sort_Order','Country_Code','Country_or_Area','Note','Type_of_Data','Group']]
Annex_2 = pd.DataFrame(Annex_2).reset_index(drop = True)

#--------------------------------------------------------------------------
# Creating Column Group in Table_2

for i in Table_2['Country_or_Area']:
    for j in Annex['Country_or_Area']:
        if i == j: 
            Table_2.loc[Table_2['Country_or_Area'] == i,'Group'] = 'Country_or_Area'

for i in Table_2['Country_or_Area']:
    for j in Annex['Major_Area']:
        if i == j: 
            Table_2.loc[Table_2['Country_or_Area'] == i,'Group'] = 'Major_Area'            

for i in Table_2['Country_or_Area']:
    for j in Annex['Region']:
        if i == j: 
            Table_2.loc[Table_2['Country_or_Area'] == i,'Group'] = 'Region'               

            
Table_2.loc[Table_2.Country_or_Area == 'WORLD','Group'] = 'WORLD'
Table_2.loc[Table_2.Country_or_Area == 'Developed regions','Group'] = 'Developed regions'
Table_2.loc[Table_2.Country_or_Area == 'Developing regions','Group'] = 'Developing regions'
Table_2.loc[Table_2.Country_or_Area == 'Least developed countries','Group'] = 'Least developed countries'
Table_2.loc[Table_2.Country_or_Area == 'Less developed regions excluding least developed countries','Group'] = 'Less developed regions'
Table_2.loc[Table_2.Country_or_Area == 'Sub-Saharan Africa','Group'] = 'Sub-Saharan Africa'


#--------------------------------------------------------------------------
# Creating Column Group in Table_3

for i in Table_3['Country_or_Area']:
    for j in Annex['Country_or_Area']:
        if i == j: 
            Table_3.loc[Table_3['Country_or_Area'] == i,'Group'] = 'Country_or_Area'

for i in Table_3['Country_or_Area']:
    for j in Annex['Major_Area']:
        if i == j: 
            Table_3.loc[Table_3['Country_or_Area'] == i,'Group'] = 'Major_Area'            

for i in Table_3['Country_or_Area']:
    for j in Annex['Region']:
        if i == j: 
            Table_3.loc[Table_3['Country_or_Area'] == i,'Group'] = 'Region'               

            
Table_3.loc[Table_3.Country_or_Area == 'WORLD','Group'] = 'WORLD'
Table_3.loc[Table_3.Country_or_Area == 'Developed regions','Group'] = 'Developed regions'
Table_3.loc[Table_3.Country_or_Area == 'Developing regions','Group'] = 'Developing regions'
Table_3.loc[Table_3.Country_or_Area == 'Least developed countries','Group'] = 'Least developed countries'
Table_3.loc[Table_3.Country_or_Area == 'Less developed regions excluding least developed countries','Group'] = 'Less developed regions'
Table_3.loc[Table_3.Country_or_Area == 'Sub-Saharan Africa','Group'] = 'Sub-Saharan Africa'


#--------------------------------------------------------------------------
# Creating Column Group in Table_4

for i in Table_4['Country_or_Area']:
    for j in Annex['Country_or_Area']:
        if i == j: 
            Table_4.loc[Table_4['Country_or_Area'] == i,'Group'] = 'Country_or_Area'

for i in Table_4['Country_or_Area']:
    for j in Annex['Major_Area']:
        if i == j: 
            Table_4.loc[Table_4['Country_or_Area'] == i,'Group'] = 'Major_Area'            

for i in Table_4['Country_or_Area']:
    for j in Annex['Region']:
        if i == j: 
            Table_4.loc[Table_4['Country_or_Area'] == i,'Group'] = 'Region'               

            
Table_4.loc[Table_4.Country_or_Area == 'WORLD','Group'] = 'WORLD'
Table_4.loc[Table_4.Country_or_Area == 'Developed regions','Group'] = 'Developed regions'
Table_4.loc[Table_4.Country_or_Area == 'Developing regions','Group'] = 'Developing regions'
Table_4.loc[Table_4.Country_or_Area == 'Least developed countries','Group'] = 'Least developed countries'
Table_4.loc[Table_4.Country_or_Area == 'Less developed regions excluding least developed countries','Group'] = 'Less developed regions'
Table_4.loc[Table_4.Country_or_Area == 'Sub-Saharan Africa','Group'] = 'Sub-Saharan Africa'

#--------------------------------------------------------------------------
# Creating Column Group in Table_5

for i in Table_5['Country_or_Area']:
    for j in Annex['Country_or_Area']:
        if i == j: 
            Table_5.loc[Table_5['Country_or_Area'] == i,'Group'] = 'Country_or_Area'

for i in Table_5['Country_or_Area']:
    for j in Annex['Major_Area']:
        if i == j: 
            Table_5.loc[Table_5['Country_or_Area'] == i,'Group'] = 'Major_Area'            

for i in Table_5['Country_or_Area']:
    for j in Annex['Region']:
        if i == j: 
            Table_5.loc[Table_5['Country_or_Area'] == i,'Group'] = 'Region'               

            
Table_5.loc[Table_5.Country_or_Area == 'WORLD','Group'] = 'WORLD'
Table_5.loc[Table_5.Country_or_Area == 'Developed regions','Group'] = 'Developed regions'
Table_5.loc[Table_5.Country_or_Area == 'Developing regions','Group'] = 'Developing regions'
Table_5.loc[Table_5.Country_or_Area == 'Least developed countries','Group'] = 'Least developed countries'
Table_5.loc[Table_5.Country_or_Area == 'Less developed regions excluding least developed countries','Group'] = 'Less developed regions'
Table_5.loc[Table_5.Country_or_Area == 'Sub-Saharan Africa','Group'] = 'Sub-Saharan Africa'


#--------------------------------------------------------------------------
# Creating Column Group in Table_6

for i in Table_6['Country_or_Area']:
    for j in Annex['Country_or_Area']:
        if i == j: 
            Table_6.loc[Table_6['Country_or_Area'] == i,'Group'] = 'Country_or_Area'

for i in Table_6['Country_or_Area']:
    for j in Annex['Major_Area']:
        if i == j: 
            Table_6.loc[Table_6['Country_or_Area'] == i,'Group'] = 'Major_Area'            

for i in Table_6['Country_or_Area']:
    for j in Annex['Region']:
        if i == j: 
            Table_6.loc[Table_6['Country_or_Area'] == i,'Group'] = 'Region'               

            
Table_6.loc[Table_6.Country_or_Area == 'WORLD','Group'] = 'WORLD'
Table_6.loc[Table_6.Country_or_Area == 'Developed regions','Group'] = 'Developed regions'
Table_6.loc[Table_6.Country_or_Area == 'Developing regions','Group'] = 'Developing regions'
Table_6.loc[Table_6.Country_or_Area == 'Least developed countries','Group'] = 'Least developed countries'
Table_6.loc[Table_6.Country_or_Area == 'Less developed regions excluding least developed countries','Group'] = 'Less developed regions'
Table_6.loc[Table_6.Country_or_Area == 'Sub-Saharan Africa','Group'] = 'Sub-Saharan Africa'

# END OF READING PROCESS: WE HAVE TABLES 1 TO 6 and supporting Tables: Annex, Notes and Content
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# Dataset cleansing step 1: Reorganizing the data in different Tables (Worksheets)
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

# melting Table_1 - the first step to create Gender and Year columns

Table_1 = pd.melt(Table_1,
                       id_vars = ['C_Sort_Order','Country_or_Area','Note','Country_Code','Type_of_Data','Group']
                      
                      )
# spliting variable - the second step to create Gender and Year columns

Table_1[['Table','Gender','Year']] = Table_1['variable'].str.split('_', expand = True)

Table_1.drop(['variable','Table'],axis = 1, inplace = True)

#Table_1.loc[Table_1['Gender'] == 'T','Gender'] = 'B' not necessary

# melting Table_2 - the first step to create Gender and Year columns

Table_2 = pd.melt(Table_2,
                       id_vars = ['C_Sort_Order','Country_or_Area','Note','Country_Code','Group']
                      
                      )
# spliting variable - the second step to create Gender and Year columns

Table_2[['Table','Gender','Year']] = Table_2['variable'].str.split('_', expand = True)

Table_2.drop(['variable','Table'],axis = 1, inplace = True)

# melting Table_3 - the first step to create Gender and Year columns

Table_3 = pd.melt(Table_3,
                       id_vars = ['C_Sort_Order','Country_or_Area','Note','Country_Code','Type_of_Data','Group']
                      
                      )
# spliting variable - the second step to create Gender and Year columns

Table_3[['Table','Gender','Year']] = Table_3['variable'].str.split('_', expand = True)

Table_3.drop(['variable','Table'],axis = 1, inplace = True)

# melting Table_4 - the first step to create Gender and Year columns

Table_4 = pd.melt(Table_4,
                       id_vars = ['C_Sort_Order','Country_or_Area','Note','Country_Code','Type_of_Data','Group']
                      
                      )
# spliting variable - the second step to create Gender and Year columns

Table_4[['Table','Gender','Year']] = Table_4['variable'].str.split('_', expand = True)

Table_4.drop(['variable','Table'],axis = 1, inplace = True)

# melting Table_5 - the first step to create Gender and Year columns

Table_5 = pd.melt(Table_5,
                       id_vars = ['C_Sort_Order','Country_or_Area','Note','Country_Code','Type_of_Data','Group']
                      
                      )
# spliting variable - the second step to create Gender and Year columns

Table_5[['Table','Gender','Year']] = Table_5['variable'].str.split('_', expand = True)

Table_5.drop(['variable','Table'],axis = 1, inplace = True)


# melting Table_6 - the first step to create Gender and Year columns

Table_6 = pd.melt(Table_6,
                       id_vars = ['C_Sort_Order','Country_or_Area','Note','Type_of_Data','Country_Code','Group']
                      
                      )
# spliting variable - the second step to create Gender and Year columns


Table_6[['Table','Gender','Year']] = Table_6['variable'].str.split('_', expand = True)

Table_6.drop(['variable','Table'],axis = 1, inplace = True)

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

# Table 6 : Breakdown into 2 Tables: E_PR and RoC

Table_6_E_PR = Table_6.loc[(Table_6.Gender == 'TT') | (Table_6.Gender == 'PR')]
Table_6_RoC = Table_6.loc[Table_6.Gender == 'RC'] 

# pivoting Table_6_E_PR to create columns Est and PR

Table_6_E_PR = pd.DataFrame(Table_6_E_PR)

Table_6_E_PR['value'] = np.where(Table_6_E_PR['value'] == '..','',Table_6_E_PR['value'])  
Table_6_E_PR['value'] = Table_6_E_PR['value'].apply(pd.to_numeric)

Table_6_PV = Table_6_E_PR.pivot_table(
    index = ['C_Sort_Order','Country_or_Area','Country_Code','Group','Year'], # 'Note','Type_of_Data',
    columns = 'Gender',
    values = 'value'
)
Table_6_PV = Table_6_PV.reset_index()

# creating Gender as 'T' since Est and PR and RoC in Table 6 are only available for both sexes

Table_6_PV['Gender'] = 'T'
Table_6_PV.rename(columns = {'TT':'Est'}, inplace = True)
Table_6_RoC = pd.DataFrame(Table_6_RoC)
Table_6_RoC['Gender'] = 'T'

# removing 00 in field Year Table_6_PV

Table_6_PV2 = Table_6_PV.assign(Year = lambda x: x.Year.str[0:4].astype(str))

#----------------------------------------------------------------------------------------------
#Ready for appending to create: Dataset_1 and Dataset_2 
#with PK: Country_Code & C_Sort_Order & Gender & Year
#with FK: 'Code' to Connect with Annex
#----------------------------------------------------------------------------------------------

#Dataset 1: 
Table_1
Table_2
Table_3
Table_4
Table_6_PV2

#Dataset 2:
Table_5
Table_6_RoC



Table_1.rename(columns = {'value':'Table_1'}, inplace = True)
Table_2.rename(columns = {'value':'Table_2'}, inplace = True)
Table_3.rename(columns = {'value':'Table_3'}, inplace = True)
Table_4.rename(columns = {'value':'Table_4'}, inplace = True)
Table_5.rename(columns = {'value':'Table_5'}, inplace = True)
Table_6_PV2.rename(columns = {'PR':'Table_6_PR','Est':'Table_6_Est'}, inplace = True)
Table_6_RoC.rename(columns = {'value':'Table_6_RoC'}, inplace = True)

# Merging Table 1 and 2

Dataset_1 = (Table_1.merge(Table_2).merge(Table_3).merge(Table_4,how = 'left').merge(Table_6_PV2,how = 'left'))
Dataset_2 = (Table_5.merge(Table_6_RoC,how = 'left'))


#Here are the files that we have so far:

Dataset_1
Dataset_2
Annex
Notes
Content


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#Datatypes and values adjustment


Notes['Note'] = Notes['Note'].astype(str)


Dataset_1['Year'] = Dataset_1['Year'].astype(str)
Dataset_1['Year'] = Dataset_1['Year'].astype(str)

Dataset_1['Table_1'] = np.where(Dataset_1['Table_1'] == '..','',Dataset_1['Table_1'])  
Dataset_1['Table_2'] = np.where(Dataset_1['Table_2'] == '..','',Dataset_1['Table_2'])  
Dataset_1['Table_3'] = np.where(Dataset_1['Table_3'] == '..','',Dataset_1['Table_3'])  
Dataset_1['Table_4'] = np.where(Dataset_1['Table_4'] == '..','',Dataset_1['Table_4'])  
Dataset_2['Table_5'] = np.where(Dataset_2['Table_5'] == '..','',Dataset_2['Table_5'])  
Dataset_2['Table_6_RoC'] = np.where(Dataset_2['Table_6_RoC'] == '..','',Dataset_2['Table_6_RoC'])  

Dataset_1['Table_1'] = Dataset_1['Table_1'].apply(pd.to_numeric)
Dataset_1['Table_2'] = Dataset_1['Table_2'].apply(pd.to_numeric)
Dataset_1['Table_3'] = Dataset_1['Table_3'].apply(pd.to_numeric)
Dataset_1['Table_4'] = Dataset_1['Table_4'].apply(pd.to_numeric)
Dataset_2['Table_5'] = Dataset_2['Table_5'].apply(pd.to_numeric)
Dataset_2['Table_6_RoC'] = Dataset_2['Table_6_RoC'].apply(pd.to_numeric)



Dataset_1['Country_Code'] = Dataset_1['Country_Code'].astype(str)
Dataset_1['Year'] = Dataset_1['Year'].astype(str)
Dataset_2['Country_Code'] = Dataset_2['Country_Code'].astype(str)
Dataset_2['Year'] = Dataset_2['Year'].astype(str)

Dataset_1.rename(columns = {'Country_Code': 'Code'}, inplace = True)
Dataset_2.rename(columns = {'Country_Code': 'Code'}, inplace = True)

Annex_2.rename(columns = {'C_Sort_Order':'Sort_Order',
                            'Country_or_Area':'Title',
                            'Country_Code': 'Code'
                            }, inplace = True)


Dataset_1['Table_2'] = Dataset_1['Table_2'] * 1000

Dataset_2 = Dataset_2.assign(Year = lambda x: x.Year.str[0:4] + '-' + x.Year.str[4:])#.astype(str)

#Dataset_1 = Dataset_1.style.format({'Table_3': '{:,.2}%'.format})

Dataset_1 = Dataset_1[['Code','Year','Gender','Table_1','Table_2','Table_3','Table_4','Table_6_PR','Table_6_Est']]

Dataset_2 = Dataset_2[['Code','Year','Gender','Table_5','Table_6_RoC']]

# spliting Type of data into 3 new columns ToD 1 to 3
Annex_2 = Annex_2[['Sort_Order','Code','Group','Title','Note','Type_of_Data']]
Annex_2[['ToD_1','ToD_2','ToD_3']] = Annex_2['Type_of_Data'].str.split(' ', expand = True)
Annex_2 = Annex_2.fillna('')
Annex_2.drop(['Type_of_Data'],axis = 1, inplace = True)

Annex = Annex_2
Annex['Code'] = Annex['Code'].astype(str)

# adding the required Columns with lables 'Yes' or 'No' for calculating aggregations in the Tables
Sub_Annex = pd.DataFrame(Sub_Annex)
Sub_Annex.rename(columns = {'Country_Code':'Code'}, inplace = True)
Sub_Annex['Code'] = Sub_Annex['Code'].astype(str)
Annex = (Annex.merge(Sub_Annex,how = 'left'))
Annex = Annex.fillna('No')
#Annex_T.to_csv('Annex_T.csv')



Notes = Notes.drop(labels=0, axis=0)




Type_of_Data = pd.DataFrame(np.array([['B','the estimates refer to the foreign-born population'],
                             ['C','The estimates refer to the foreign citizens'],
                             ['R','the number of refugees, as reported by UNHCR, were added to the estimate of international migrants'],
                             ['I','Estimates for countries or areas having no data on the number of international migrants were obtained by imputation indicated by'],
                            ]),
                    columns=['ToD', 'Desc'])

Content.loc[Content.Variable == 'Table 1','Variable'] = 'Table_1'
Content.loc[Content.Variable == 'Table 2','Variable'] = 'Table_2'
Content.loc[Content.Variable == 'Table 3','Variable'] = 'Table_3'
Content.loc[Content.Variable == 'Table 4','Variable'] = 'Table_4'
Content.loc[Content.Variable == 'Table 5','Variable'] = 'Table_5'


new_row = {'Variable':'Table_6_PR','Desc':'Refugees as a percentage of the international migrant stock'}
#Content = Content.append(new_row, ignore_index=True)
new_row = {'Variable':'Table_6_Est','Desc':'Estimated refugee stock at mid-year (both sexes)'}
#Content = Content.append(new_row, ignore_index=True)
new_row = {'Variable':'Table_6_RoC','Desc':'Annual rate of change of the refugee stock'}
#Content = Content.append(new_row, ignore_index=True)
new_row = {'Variable':'ToD','Desc':'Type of Data'}
#Content = Content.append(new_row, ignore_index=True)

Content = Content.drop(labels=5, axis=0)
Content = pd.DataFrame(Content).reset_index(drop = True)


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
# End of Step 1. Here are output Tables at the end of this step
Dataset_1
Dataset_2
Annex
Notes
Type_of_Data
Content
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------



#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# Dataset cleansing Step 2: Eliminating calculated rows and columns
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

Dataset_1.drop(['Table_3','Table_4','Table_6_PR'],axis = 1, inplace = True)


Annex_C = Annex.loc[lambda Annex: Annex['Group'] == 'Country_or_Area']

Dataset_1 = pd.merge(Dataset_1,
                      Annex_C,
                      left_on = 'Code',
                      right_on = 'Code',
                      how = 'inner'
                     )



Dataset_1_1 = Dataset_1[['Code','Year','Gender','Table_1']]
Dataset_1_1 = Dataset_1_1.fillna('')

Dataset_1_2 = Dataset_1[['Code','Year','Gender','Table_2','Table_6_Est']]
Dataset_1_2 = Dataset_1_2.fillna('')

Dataset_1_1 = Dataset_1_1.loc[(Dataset_1.Gender == 'M') | (Dataset_1.Gender == 'F')]


#Dataset_1_2 = Dataset_1_2.loc[Dataset_1.Gender == 'T']


Dataset_2_1 = pd.merge(Dataset_2,
                      Annex_C,
                      left_on = 'Code',
                      right_on = 'Code',
                      how = 'inner'
                     )

Dataset_2_1 = Dataset_2_1[['Code','Year','Gender','Table_5','Table_6_RoC']]
Dataset_1_2 = Dataset_1_2.fillna('')


Annex_O = Annex.loc[lambda Annex: Annex['Group'] != 'Country_or_Area']

Dataset_2_2 = pd.merge(Dataset_2,
                      Annex_O,
                      left_on = 'Code',
                      right_on = 'Code',
                      how = 'inner'
                     )

Dataset_2_2 = Dataset_2_2[['Code','Year','Gender','Table_5','Table_6_RoC']]
Dataset_2_2 = Dataset_2_2.fillna('')


Dataset_1_1['Code'] = Dataset_1_1['Code'].apply(pd.to_numeric)
Dataset_1_2['Code'] = Dataset_1_2['Code'].apply(pd.to_numeric)
Dataset_2_1['Code'] = Dataset_2_1['Code'].apply(pd.to_numeric)
Dataset_2_2['Code'] = Dataset_2_2['Code'].apply(pd.to_numeric)

Dataset_1_1 = Dataset_1_1.sort_values(['Code','Year','Gender'], ascending = (True, True, True))
Dataset_1_2 = Dataset_1_2.sort_values(['Code','Year','Gender'], ascending = (True, True, True))
Dataset_2_1 = Dataset_2_1.sort_values(['Code','Year','Gender'], ascending = (True, True, True))
Dataset_2_2 = Dataset_2_2.sort_values(['Code','Year','Gender'], ascending = (True, True, True))
Annex = Annex.sort_values(['Sort_Order','Group','Code'], ascending = (True, True, True))


Dataset_1_1['Code'] = Dataset_1_1['Code'].astype(str)
Dataset_1_2['Code'] = Dataset_1_2['Code'].astype(str)
Dataset_2_1['Code'] = Dataset_2_1['Code'].astype(str)
Dataset_2_2['Code'] = Dataset_2_2['Code'].astype(str)



#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
# End of Step 2. Here are the output Tables at the end of this step
Dataset_1_1
Dataset_1_2
Dataset_2_1
Dataset_2_2
Annex
Notes
Type_of_Data
Content
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
# End of Data Cleaning process. Final Tables are ready to export to Excel
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

with pd.ExcelWriter('UN_MigrantStockTotal_2015_Output.xlsx') as writer:  
    Dataset_1_1.to_excel(writer, sheet_name='Dataset_1_1', index = False)
    Dataset_1_2.to_excel(writer, sheet_name='Dataset_1_2', index = False)
    Dataset_2_1.to_excel(writer, sheet_name='Dataset_2_1', index = False)
    Dataset_2_2.to_excel(writer, sheet_name='Dataset_2_2', index = False)
    Annex.to_excel(writer, sheet_name='Annex', index = False)
    Notes.to_excel(writer, sheet_name='Notes', index = False)
    Content.to_excel(writer, sheet_name='Content', index = False)
    Type_of_Data.to_excel(writer, sheet_name='Type_of_Data', index = False)


print("The output Excel file is generated as UN_MigrantStockTotal_2015_Output")    
print("************This is the end of Mid-Term project: Hamid Parsazadeh****************")