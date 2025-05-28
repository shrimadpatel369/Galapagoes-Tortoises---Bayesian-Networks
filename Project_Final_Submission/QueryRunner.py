from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import PC
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display,Image




class FemaleModel():
    def __init__(self,data):
        self.data = data
        try:
            data['eggs detected'].astype(int)
        except Exception as e:
            print("There was an error in the Dataset Preprocessing : " + str(e))
            pass
        
        

        edges = [('Year', 'Body condition Index'),\
                ('Elevation m', 'eggs detected'),\
                ('Body condition Index', 'eggs detected')]
        

        self.bayesian_model = BayesianNetwork(edges)

        # Create a maximum likelihood estimator

        # Fit the model to the data
        self.bayesian_model.fit(self.data, estimator=MaximumLikelihoodEstimator)
        print("Female Model Ready")

        # Initialize inference on the Bayesian network
        self.infer = VariableElimination(self.bayesian_model)


    def categorizeElevation(self,x):
        try:
            x = float(x)
            if 0 < x < 100:
                return 0
            elif 100 <= x < 200:
                return 1
            elif 200 <= x < 300:
                return 2
            else:
                return 3
        except:
            print("Please enter a valid value (Should be an integer or float)")
        
        
    def categorizeBCI(self,x):
        try:
            x = float(x)
            if x < 0.8:
                return 0
            elif   0.8 <= x < 1:
                return 1
            elif 1 <= x < 1.2:
                return 2
            elif 1.2 <= x < 1.4:
                return 3
            elif 1.4 <= x < 1.6:
                return 4
            else:
                return 5
        except:
            print("Please enter a valid value (Should be an integer or float)")
            

        
        


        
    def convertInput(self,bci,elevation):
        elevation = self.categorizeElevation(elevation)
        bci = self.categorizeBCI(bci)

        return bci,elevation
    
    def showModel(self):
        display(Image((nx.drawing.nx_pydot.to_pydot(self.bayesian_model)).create_png()))


    def process(self,bci,elevation):
        bci,elevation = self.convertInput(bci=bci,elevation=elevation)
        query_result = self.infer.query(
            variables=["eggs detected"],  # Replace with your target variable's name

            evidence={'Body condition Index':bci,
                      'Elevation m':elevation
                      }  # Observed values
        )
        return query_result
        df = pd.DataFrame(query_result.values,columns=['x'])
        return df

        



class EggModel:
    def __init__(self, data):
        self.data = data

        # Convert categorical variables to numerical variables
        try:
            data['Temperature'] = data['Temperature'].apply(lambda x: EggModel.categorizeTemp(x))
            data['NDVI'] = data['NDVI'].apply(lambda x: EggModel.categorizeNDVI(x))
            data['Rainfall'] = data['Rainfall'].apply(lambda x: EggModel.categorizeRain(x))
            data['Nest area'] = data['Nest area'].apply(lambda x: EggModel.convertNestArea(x))
        except Exception as e:
            print(f"Error converting categorical variables: {e}")
            pass

        self.data = data
        

        edges = [('Temperature', 'Health Index'), ('Temperature', 'NDVI'), ('Temperature', 'Rainfall'), ('NDVI', 'Health Index'), ('Rainfall', 'NDVI'), ('Nest area', 'Temperature'), ('Nest area', 'Rainfall'), ('Nest area', 'NDVI')]

        self.bayesian_model = BayesianNetwork(edges)

        # Create a maximum likelihood estimator

        # Fit the model to the data
        self.bayesian_model.fit(self.data, estimator=MaximumLikelihoodEstimator)
        print("Egg Model Ready")

        # Initialize inference on the Bayesian network
        self.infer = VariableElimination(self.bayesian_model)



    def showModel(self):
        display(Image((nx.drawing.nx_pydot.to_pydot(self.bayesian_model)).create_png()))

    def convertInput(self,ndvi,temp):
        ndvi = float(ndvi)
        temp = float(temp)
        ndvi = EggModel.categorizeNDVI(ndvi)
        temp = EggModel.categorizeTemp(temp)
        return ndvi, temp
        
    def convertNestArea(x):
        arr = str(x).split()
        try : 
            if arr[2] == 'Lower':
                return 0
            elif arr[2] == 'Upper' :
                return 2
            else:
                return 1
        except:
            return int(x)
    

    # <24:0 24-25:1 25-26:2 26-27:3 >27:4
    def categorizeTemp(temp):
        if temp<24:
            return 0
        elif 24<=temp<25:
            return 1
        elif 25<=temp<26:
            return 2
        elif 26<=temp<27:
            return 3
        else:
            return 4
        
    # <5000:0 5000-6000:1 6000-7000:2 7000-8000:3 >8000:4
    def categorizeNDVI(ndvi):
        ndvi = float(ndvi)
        if ndvi<5000:
            return 0
        elif 5000 <= ndvi<6000:
            return 1
        elif 6000 <= ndvi<7000:
            return 2
        elif 7000 <= ndvi < 8000:
            return 3
        else:
            return 4

    # 0-7:0 5000-6000:1 6000-7000:2 7000-8000:3 >8000:4
    def categorizeRain(rain):
        rain = float(rain)

        if 0 <= rain<7:
            return 0
        elif 7 <= rain<14:
            return 1
        elif 14 <= rain < 21:
            return 2
        elif 21 <= rain < 28:
            return 3
        else:
            return 4
        
    def process(self,ndvi,temp):
        ndvi,temp = self.convertInput(ndvi=ndvi,temp=temp)
        query_result = self.infer.query(
            variables=["Health Index"],  # Replace with your target variable's name

            evidence={'NDVI':ndvi,
                      'Temperature':temp
                      }  # Observed values
        )
        return query_result
        df = pd.DataFrame(query_result.values,columns=['x'])
        return df



class HachlingModel:
    def __init__(self, df):
        self.df = df  

        global categories
        categories = df['Nesting area'].astype('category').cat.categories

        # Display categories with their corresponding codes
        self.category_codes = {category: code for code, category in enumerate(categories)}
        
        print(self.category_codes)



        df.dropna(inplace=True)
        df.reset_index(drop=True,inplace=True)


        

        try:
            
            df['Mean_Temp C'] = df['Mean_Temp C'].apply(lambda x: HachlingModel.categorizeTemp(x))
            df['Nesting area'] = df['Nesting area'].apply(lambda x: HachlingModel.categorizeNesting(x))
            df['NDVI'] = df['NDVI'].apply(lambda x: HachlingModel.categorizeNDVI(x))
            df['Rain cm'] = df['Rain cm'].apply(lambda x: HachlingModel.categorizeRain(x))
            df['Weight grams'] = df['Weight grams'].apply(lambda x: HachlingModel.categorizeWeight(x))
            df['Initial_Mass g'] = df['Initial_Mass g'].apply(lambda x: HachlingModel.categorizeWeight(x))
            df['Length mm'] = df['Length mm'].apply(lambda x: HachlingModel.categorizeLength(x))
            df['Initial_length mm'] = df['Initial_length mm'].apply(lambda x: HachlingModel.categorizeLength(x))
            df['days_alive'] = df['days_alive'].apply(lambda x: HachlingModel.categorize_alive(x))
            # df['%_mass_change_per_day'] = df['%_mass_change_per_day'].apply(lambda x: HachlingModel.categorizeMassChange(x))
            # df['%Length_change_per_day'] = df['%Length_change_per_day'].apply(lambda x: HachlingModel.categorizeLengthChange(x))
            # df['interval_days'] = df['interval_days'].apply(lambda x: HachlingModel.categorizeInterval(x))
            # df.drop(columns=['%_mass_change_per_day','%Length_change_per_day','interval_days'],inplace=True)
            self.df = df
        except Exception as e:
            print("There was an error while trying to categorize the data : ",e)

            pass

        edges = [('Year', 'NDVI'), ('Year', 'Nesting area'), ('Year', 'Rain cm'), ('Year', 'Mean_Temp C'), ('NDVI', 'Mean_Temp C'), ('NDVI', 'days_alive'), ('Nesting area', 'Rain cm'), ('Nesting area', 'NDVI'), ('Nesting area', 'days_alive'), ('Nesting area', 'Mean_Temp C'), ('Rain cm', 'Mean_Temp C'), ('Rain cm', 'NDVI'), ('Weight grams', 'days_alive'), ('Initial_Mass g', 'days_alive'), ('Initial_Mass g', 'Weight grams'), ('Initial_length mm', 'days_alive'), ('Initial_length mm', 'Length mm'), ('Length mm', 'days_alive')]

        self.bayesian_model = BayesianNetwork(edges)

        # Create a maximum likelihood estimator

        # Fit the model to the data
        self.bayesian_model.fit(self.df, estimator=MaximumLikelihoodEstimator)
        print("Hatchling Model Ready")

        # Initialize inference on the Bayesian network
        self.infer = VariableElimination(self.bayesian_model)

    def showModel(self):
        display(Image((nx.drawing.nx_pydot.to_pydot(self.bayesian_model)).create_png()))

    def convertInput(self,ndvi,nesting,initialMass,initialLength,length,weight):
        ndvi = HachlingModel.categorizeNDVI(ndvi)
        nesting = HachlingModel.categorizeNesting(nesting)
        initialMass = HachlingModel.categorizeWeight(initialMass)
        initialLength = HachlingModel.categorizeLength(initialLength)
        length = HachlingModel.categorizeLength(length)
        weight = HachlingModel.categorizeWeight(weight)
        return ndvi, nesting, initialMass, initialLength, length, weight
        

    def getFunctions(self):
        return [HachlingModel.categorizeTemp, HachlingModel.categorizeNDVI, HachlingModel.categorizeRain, HachlingModel.categorizeWeight, HachlingModel.categorizeWeight, HachlingModel.categorizeLength, HachlingModel.categorizeLength, HachlingModel.categorize_alive, HachlingModel.categorizeMassChange, HachlingModel.categorizeLengthChange, HachlingModel.categorizeInterval,HachlingModel.showModel]
    
    def categorizeNesting(nesting):
        category_codes = {category: code for code, category in enumerate(categories)}
        return category_codes[nesting]
    def categorizeTemp(temp):
        if temp<24:
            return 0
        elif 24<=temp<25:
            return 1
        elif 25<=temp<26:
            return 2
        elif 26<=temp<27:
            return 3
        else:
            return 4
        
    # <5000:0 5000-6000:1 6000-7000:2 7000-8000:3 >8000:4
    def categorizeNDVI(ndvi):
        ndvi = float(ndvi)
        if ndvi<5000:
            return 0
        elif 5000 <= ndvi<6000:
            return 1
        elif 6000 <= ndvi<7000:
            return 2
        elif 7000 <= ndvi < 8000:
            return 3
        else:
            return 4

    # 0-7:0 5000-6000:1 6000-7000:2 7000-8000:3 >8000:4
    def categorizeRain(rain):
        rain = float(rain)

        if 0 <= rain<7:
            return 0
        elif 7 <= rain<14:
            return 1
        elif 14 <= rain < 21:
            return 2
        elif 21 <= rain < 28:
            return 3
        else:
            return 4
        
    def categorizeWeight(w):
        w = float(w)

        if 0 <= w<1000:
            return 0
        elif 1000 <= w< 2000:
            return 1
        else:
            return 2

    def categorizeLength(l):
        l = float(l)

        if 0 <= l <100:
            return 0
        elif 100 <= l < 200:
            return 1
        else:
            return 2

    def categorize_alive(days):
        days = float(days)

        return int(days//365)

    def categorizeMassChange(change):
        change = round(change,4)
        if change==0.0:
            return 1
        elif change<0:
            return 0
        else:
            return 2
    def categorizeLengthChange(change):
        change = round(change,4)
        if change <= 0.25:
            return 0
        else:
            return 1

    def categorizeInterval(days):
        days = round(days,4)
        if days < 40:
            return 0
        else:
            return 1 


    def process(self,ndvi,nesting,length,initialLength,weight,initialMass):
        ndvi,nesting,length,initialLength,weight,initialMass = self.convertInput(ndvi=ndvi,nesting=nesting,length=length,initialLength=initialLength,weight=weight,initialMass=initialMass)
        query_result = self.infer.query(
            variables=["days_alive"],  # Replace with your target variable's name

            evidence={"Initial_length mm": initialLength,
                      'Length mm':length,
                      'Initial_Mass g':initialMass,
                      'Weight grams':weight,
                      'NDVI':ndvi,
                      'Nesting area':nesting
                      }  # Observed values
        )
        return query_result
        df = pd.DataFrame(query_result.values,columns=['x'])
        return df
   