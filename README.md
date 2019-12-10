# RadarChallenge2020 team
RadarChallenge 2020 @ chongqi, Nov, 2020


### Submission Styles

- radar2020_testSubmission.zip

		- classification_results.txt
		- generalisation_results.txt
	

  
- classification task - format in  **classification_results.txt**
  
  	`` <spectrumDiagram classification name> <class_name> <confidence>  ``
  		
  	''example1: ".txt"
               1_December_2017 Dataset1P36A01R01 1 0.92	       
  	       1_December_2017 Dataset1P36A01R01 2 0.98	       
  	       1_December_2017 Dataset1P36A01R01 3 0.43	       
  	       1_December_2017 Dataset1P36A01R01 4 0.20	       
  	       1_December_2017 Dataset1P36A01R01 5 0.00
	       1_December_2017 Dataset1P36A01R01 6 0.00 
 ### class name vs numbering
 - class 1: walk back/forth
 - class 2: sit on a chair
 - class 3: stand up
 - class 4: bend down
 - class 5: bend up
 - class 6: stand and drink from cup
   	Tips:
  		
  	- make sure seperated the coloum using a space
  
- generalisation tasks - format in **generalisation_results.txt**

  `` <spectrumDiagram classification name> <class_name> <confidence>  ``
  		
  	example1: ".txt'
	         gener_spectrum1 1 0.92 
		 gener_spectrum1 2 0.98
		 gener_spectrum1 3 0.43
		 gener_spectrum1 4 0.20
		 gener_spectrum1 5 0.00
		 gener_spectrum1 6 0.00 
 ### class name vs numbering
 - class 1: walk back/forth
 - class 2: sit on a chair
 - class 3: stand up
 - class 4: bend down
 - class 5: bend up
 - class 6: stand and drink from cup

Tips:
  		
  	- make sure you used generlisation task spectrums
  	
	
### Evaluation Scoring

1. **Radar Challenge Classification Task**
	- average AP mAP=1/N  ∑[AP]_i
	*Refer to paper XXXX for understand how to caculate AP 
2. **Radar Challenge Generalization Task**
	- Deviation score per class above or below tolerance (+/-5%) will be reported
		**Highest mAP with lowest deviation score will be declared winner of this sub-challenge***

	*For example: Lets say tolerance is 10%, then if your algorithm in classification gives an mAP/class of 30% then your generalization should be with in the tolerance range, i.e., 27%<=mAP/class<=33%, in this scenario your deviation will be zero. However, anything below or above will be penalized. Lets say if your algorithm scores 25% on generalization data then your deviation will be 2% which will be reported.*
	
3. **Final Score**
	- Final score:   0.6 * mAPd + 0.4 * mAPg 
	

  		
  	
