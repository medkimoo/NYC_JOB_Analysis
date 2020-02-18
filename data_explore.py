import pandas as pd

class Nyc_job_read():
    
    def __init__(self, filename):
        self.df           = pd.read_csv(filename,
                                        dtype={"Salary Range From": "float64",
                                               "Salary Range To": "float64"})
        self.df           = pd.DataFrame(self.df)
        
        # drop Columns       
        self.df = self.df.drop(['Posting Type', '# Of Positions',
       'Civil Service Title', 'Title Code No', 'Level',
       'Full-Time/Part-Time indicator', 'Additional Information', 'To Apply', 
       'Hours/Shift', 'Work Location 1', 'Work Location', 'Division/Work Unit', 'Job Description',
       'Recruitment Contact', 'Residency Requirement', 'Posting Date',
       'Post Until', 'Posting Updated', 'Process Date'], axis = 1)
        
        # delete nan lines
        self.df           = self.df.dropna()
        
        # Drop duplicate
        self.df = self.df.drop_duplicates(subset = 'Job ID')    
        
        # Calculate Salary average
        self.df['Salary avr'] = (self.df['Salary Range From'] + self.df['Salary Range To']) / 2
        self.df['Frequency'] = self.df['Salary Frequency'].apply(lambda x: 251 if x == 'Daily' else (8 * 251 if x == 'Hourly' else 1))
        self.df['Salary avr annual'] = self.df['Salary avr'] * self.df['Frequency'] 
        self.df = self.df.drop(['Salary avr', 'Frequency', 'Salary Range From', 'Salary Range To', 'Salary Frequency'], axis = 1)
        
        # Set Job ID as index 
        self.df = self.df.set_index('Job ID')
        

if __name__ == "__main__":
        
        data = Nyc_job_read('./Data/Jobs.csv')
        print("\n> Le nombre de lignes dans le dataset est: " , len(data.df), "lignes")
        print("\n> Affichage de 5 premières lignes du dataset:")
        print("\n", data.df.head(5))
        print("\n> Affichage des features:")
        print("\n", list(data.df.columns))
        