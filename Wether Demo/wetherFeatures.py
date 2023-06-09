

class DataFeature():
  def __init__(self):
    self.data={}
    
    self.feature=[]
  def get_weatherFeatures(self,weatherdata):
    data= open(weatherdata,'r').readlines()
    for D in data:
      
      print(len(D.split(',')))
   
 
dataset=DataFeature()
 
data='data/weather.csv'
dataset.get_weatherFeatures(data)
 