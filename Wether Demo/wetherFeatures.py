import pandas as pd



class WeatherData():
  def __init__(self ):
    self.data=pd.read_csv('data/weather.csv')
    self.colums=self.data.columns.tolist()
    print(self.colums)
  def Features(self):
    features=[]
    for D in self.colums:
      d=self.data[D]
      dd=[]
      for dat in d.tolist() :
        if dat=="Yes" or dat=="N":
          dat=1
          dd.append(dat)
       
          
        if dat=="No":
          dat=0
          dd.append(dat) 
        if dat=="S":
          dat=4
          dd.append(dat)
        if dat=="E":
          dat=2
          dd.append(dat) 
        if dat=="W":
          dat=3
          dd.append(dat)
        if dat=="NW":
          dat=13
          dd.append(dat)
        if dat=="NE":
          dat=12
          dd.append(dat)
        if dat=="SW":
          dat=43
        if dat=="SE":
          dat=42
          dd.append(dat)
        if dat=="NNE":
          dat=112
          dd.append(dat)
        if dat=="NNW":
          dat=113
          dd.append(dat)
        if dat=="SSE":
          dat=442
          dd.append(dat)
        if dat=="SSW":
          dat=443
          dd.append(dat)
        if dat=="ESE":
          dat=242
          dd.append(dat)
        if dat=="ENE":
          dat=212
          dd.append(dat)
        if dat=="WNW":
          dat=313
          dd.append(dat)
        if dat=="WSW":
          dat=343
          dd.append(dat)
        else:
            dd.append(dat)
          
          
     
   
      
      
      features.append(dd)
    Y=features[21]
    X=features[:21]
      
    return X,Y 
      
    
data=WeatherData()
print(data.Features())
