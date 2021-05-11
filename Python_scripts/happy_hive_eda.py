import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

#loading the data
df = pd.read_csv('data/Manual.csv')

#making a copy of the data
df_copy = df.copy(deep=True)

#correction functions
def spell_check(row):

    '''spell check for the educaiton column'''

    if row == 'computer scince ' or row =='software eng.':
        row = 'computer science'

    if row == 'engineering ':
        row = 'engineering'

    if row == 'scince' or row == 'scinece':
        row = 'science'
    
    return row

def plot_dist(data, column, x_label, y_label):

    '''Ploting the distribution for the column'''

    x = data[column].values
    fig,ax = plt.subplots(figsize=(8,6))
    ax.set(xlabel=x_label, ylabel=y_label)
    ax = sns.distplot(x,bins=15)

def bar_plot(values, labels,x_label,y_label,title):

    '''Bar plot for the column'''

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([0,0,1,1])
    ax.bar(labels,values)
    plt.xlabel(x_label,fontsize=14)
    plt.ylabel(y_label,fontsize=14)
    plt.title(title,fontsize=14)
    plt.show()
    
def pie_chart(values,labels,title,sub=None):

    '''Pie chart for the values'''

    fig1, ax1 = plt.subplots(figsize=(8,6))
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    ax1.pie(values, explode=explode, 
            labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90
            )

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title(title,fontsize=14)

    plt.show()


def main():

    #cleaning the data

    df_copy['education'] = df_copy['education'].appy(spell_check)

    #if the manager is 1 then senior should be 1 in the data
    df_copy.loc[df_copy['manager'] == 1, 'senior'] = 1

    #dividing into male and female data
    df_copy_male = df_copy.loc[df_copy['gender'] == 'M']
    df_copy_female = df_copy.loc[df_copy['gender'] == 'F']

    #plot the distribution for the male and female data
    plot_dist(df_copy_male,'age','Male Age','Density')
    plot_dist(df_copy_female,'age','Female Age','Density')

    #drop the columns we have not filled
    df_copy.drop(['level_education','years_of_experience'], axis=1, inplace=True)

    #number of men and women in the senior levels

    #number of men and women in senior levels
    senior_male_count = df_copy_male.loc[df_copy_male['senior'] == 1]['age'].count()
    senior_female_count = df_copy_female.loc[df_copy_female['senior'] == 1]['age'].count()

    #average age of men and women in senior levels
    senior_male_avg = df_copy_male.loc[df_copy_male['senior'] == 1]['age'].mean()
    senior_female_avg = df_copy_female.loc[df_copy_female['senior'] == 1]['age'].mean()

    print('number of senior male and average age for male senior:', senior_male_avg,",", senior_male_count)
    print('number of senior female and average age for female senior:',senior_female_avg,",",senior_female_count)
    print('-----------'*10)
    bar_plot([senior_male_count,senior_female_count],['male','female'],'gender','count','Number of Senior Positions')


    #number of men and women in managerial position
    manager_male_count = df_copy_male.loc[df_copy_male['manager'] == 1]['age'].count()
    manager_female_count = df_copy_female.loc[df_copy_female['manager'] == 1]['age'].count()

    #average age of men and women in managerial levels
    manager_male_avg = df_copy_male.loc[df_copy_male['manager'] == 1]['age'].mean()
    manager_female_avg = df_copy_female.loc[df_copy_female['manager'] == 1]['age'].mean()

    print('number of manager male and average age for male manager:', manager_male_avg,",", manager_male_count)
    print('number of manager female and average age for female manager:',manager_female_avg,",",manager_female_count)

    print('----------'*10)
    bar_plot([manager_male_count,manager_female_count],['male','female'],'gender','count','Number of Manager Positions')

    #average duration of men and women in the compnay
    average_duration_m = df_copy.loc[df['gender'] == 'M']['duration'].mean()
    average_duration_f = df_copy.loc[df['gender'] == 'F']['duration'].mean()

    print('average duration for men:',average_duration_m)
    print('average duratino for women:',average_duration_f)
    print('-------'*10)
    bar_plot([average_duration_m,average_duration_f],['male','female'],
            'gender','count','Average duration')


    #average duration for managerial positons
    average_duration_m_manager = df_copy_male[df_copy_male['manager'] == 1]['duration'].mean()
    average_duration_f_manager = df_copy_female[df_copy_female['manager'] == 1]['duration'].mean()

    print('average duration for male managers:', average_duration_m_manager)
    print('average duration for female managers:', average_duration_f_manager)
    print('--------'*10)
    average_duration_m_manager = df_copy_male[df_copy_male['manager'] == 1]['duration'].mean()
    average_duration_f_manager = df_copy_female[df_copy_female['manager'] == 1]['duration'].mean()

    bar_plot([average_duration_m_manager,average_duration_f_manager],
            ['male','female'],'gender','count','Average Duration for Managers')


    #number of women who joined recently and gained managerial position

    no_manager_less_five_m = df_copy_male[(df_copy_male['manager'] == 1) & 
                                        (df_copy_male['duration']<=5) &
                                        (df_copy_male['no_company_10'] <=2)]['age'].count()

    no_manager_less_five_f = df_copy_female[(df_copy_female['manager'] == 1) & 
                                        (df_copy_female['duration']<=5) &
                                        (df_copy_female['no_company_10'] <=2)]['age'].count()

    bar_plot([no_manager_less_five_m,no_manager_less_five_f],['male','female'],
            'gender','count',
            'Candidates Reached Managerial Position With no Prior Work Experience')

    
    #number of women who joined recently and gained managerial position

    no_manager_less_five_m_pos = df_copy_male[(df_copy_male['manager'] == 1) & 
                                        (df_copy_male['no_job_positions'] == 0) &
                                        (df_copy_male['no_company_10'] == 0)]['age'].count()

    no_manager_less_five_f_pos = df_copy_female[(df_copy_female['manager'] == 1) & 
                                        (df_copy_female['no_job_positions']==0) &
                                        (df_copy_female['no_company_10'] == 0)]['age'].count()

    bar_plot([no_manager_less_five_m,no_manager_less_five_f],['male','female'],'gender',
            'count',
            'Candidates Reached Managerial Position With no Prior Work Experience')

    
    #number of women who joined recently and gained managerial position

    no_manager_less_five_m = df_copy_male[(df_copy_male['manager'] == 1) & 
                                        (df_copy_male['duration']<=10)]['age'].count()

    no_manager_less_five_f = df_copy_female[(df_copy_female['manager'] == 1) & 
                                        (df_copy_female['duration']<=10)]['age'].count()

    bar_plot([no_manager_less_five_m,no_manager_less_five_f],['male','female'],
            'gender','count',
            'Candidates Reached Managerial Position With no Prior Work Experience')

    #pie chart for the education background
    educ_values = df_copy.groupby('education')['name'].count()
    labels = values.index[:].tolist()
    val = values.values.tolist()
    pie_chart(val,labels,'Education Percentage')

    #Educaitonal background for male and female in shopify 
    educ_male = df_copy_male.groupby('education')['name'].count()
    labels_male = educ_male.index[:]
    values_male = educ_male.values[:]

    pie_chart(values_male,labels_male,"Male Education")

    educ_female = df_copy_female.groupby('education')['name'].count()
    labels_female = educ_female.index[:]
    values_female = educ_female.values[:]

    pie_chart(values_female,labels_female,"Female Education")

    #Education background for male and female managers

    educ_male_manager = df_copy_male.loc[df_copy_male['manager'] == 1].groupby('education')['name'].count()
    manager_value_male = educ_male_manager.values[:]
    manager_index_male = educ_male_manager.index[:]

    educ_female_manager = df_copy_female.loc[df_copy_female['manager'] == 1].groupby('education')['name'].count()
    manager_value_female = educ_female_manager.values[:]
    manager_index_female = educ_female_manager.index[:]


    pie_chart(manager_value_male,manager_index_male,"Male Manager Education")
    pie_chart(manager_value_female,manager_index_female,"Female Manager Education")



if __name__=="__main__":
    main()
