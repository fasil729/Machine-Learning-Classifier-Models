

class DataFeature():
  def __init__(self):
    self.data={}
    
    self.feature=[]
  def get_weatherFeatures(self,weatherdata):
    data= open(weatherdata,'r').readlines()
    for D in data:
      
      print(D.split(',')[:21])
  def labelOfWeather(self,weatherdata):
    data= open(weatherdata,'r').readlines()
    label=[]
    for D in data:
      
      label.append(D.split(',')[21])
    label.pop(0)
   
    return label
dataset=DataFeature()
 
data='data/weather.csv'
dataset.get_weatherFeatures(data)
print(dataset.labelOfWeather(data))
 