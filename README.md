## Vietnamese sign lagnuage recognition using MHI and CNN

This is a model to classify Vietnamese sign language using Motion history image (MHI) algorithm and CNN. 
We have 6 classes for most basic vocabolary disabled people need in daily life :

+ Uống (Drink)
+ Đói (Hungry)
+ Vui (Happy)
+ Giận (Angry)
+ Tôi (Me)
+ and class None for indicating no actions. 


<img src="https://user-images.githubusercontent.com/57822898/149609119-0b23e938-6dec-4148-847a-603704895511.png" width=70% height=70%>
<img src="https://user-images.githubusercontent.com/57822898/149609149-3da23c1a-e791-451d-b402-ec7e8a2cbf95.png" width=70% height=70%>
<img src="https://user-images.githubusercontent.com/57822898/149609126-e0c1a346-6adf-4e84-9b97-40db4a0a06fd.png" width=70% height=70%>
<img src="https://user-images.githubusercontent.com/57822898/149609137-46ea406f-6ff5-4500-8f7c-999b662314d3.png" width=70% height=70%>
<img src="https://user-images.githubusercontent.com/57822898/149609124-849727d8-f14a-40ad-b80c-6aa2e453715a.png" width=70% height=70%>
<img src="https://user-images.githubusercontent.com/57822898/149609096-197e01bd-e1e2-4eb0-afb7-f47c7f487251.png" width=70% height=70%>

## Motion history images (MHI)
First proposed by [A.Bobick et.al](https://ieeexplore.ieee.org/document/910878) as a method to capture both spatial and temporal information of an action. Here some demonstration. 


<img src="https://user-images.githubusercontent.com/57822898/149686938-62984bcf-1dd3-4349-a062-a883df4deb0e.gif" width=50% height=50%>

This algorithm conveniently enables CNN to recognize actions, not just still images. Since sign languages are consisted of dynamic gesutres, it's perfect for this project.



### Requirements
  Python3</br>
  Numpy</br>
  Tensorflow</br>
  Keras</br>
  opencv</br>
  Matplotlib</br>
  Seaborn</br>
 

