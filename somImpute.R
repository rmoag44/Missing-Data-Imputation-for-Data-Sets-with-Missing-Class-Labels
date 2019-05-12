require(CoImp)
require(pROC)
require(glmnet)
require(kohonen)
require(MASS)
require(randomForest)


hd = read.csv("heartDisease.csv")
bc = read.csv("wdbc.csv")
d = read.csv("diabetes.csv")



errorSOMGLM = matrix(0,nrow = 3,ncol = 9)
errorSOMLDA = matrix(0,nrow = 3,ncol = 9)
errorSOMRF = matrix(0,nrow = 3,ncol = 9)
errorSOMGLMDel = matrix(0,nrow = 3,ncol = 9)
errorSOMLDADel = matrix(0,nrow = 3,ncol = 9)
errorSOMRFDel = matrix(0,nrow = 3,ncol = 9)


pb <- txtProgressBar(min = 1, max = 100, style = 3)
for (i in 1:100) 
{
  set.seed(i)
  setTxtProgressBar(pb, i)
  for (j in 1:9)
  {
    # Simulate appropriate percentage of missing data
    
    hdMissing = MCAR(as.matrix(hd),perc.miss = 0.05 * j,setseed = i)@db.missing
    bcMissing = MCAR(as.matrix(bc),perc.miss = 0.05 * j,setseed = i)@db.missing
    dMissing = MCAR(as.matrix(d),perc.miss = 0.05 * j,setseed = i)@db.missing
    
    imputedHd1 = hdMissing
    imputedBc1 = bcMissing
    imputedD1 = dMissing
    
    # Set grid dimension
    
    mygridH = somgrid(9,10,"hexagonal")
    mygridB = somgrid(10,12,"hexagonal")
    mygridD = somgrid(10,14,"hexagonal")
    
    # Train the SOM for each data set
    
    hd.som = supersom(hdMissing,mygridH,maxNA.fraction = 1)
    bc.som = supersom(bcMissing,mygridB,maxNA.fraction = 1)
    d.som = supersom(dMissing,mygridD,maxNA.fraction = 1)

    #Impute missing values with their corresponding values in their BMU. 
    
    for (k in 1:297) {
      for (l in 1:14) {
        imputedHd1[k,l] = ifelse(is.na(imputedHd1[k,l]),as.data.frame(hd.som$codes)[hd.som$unit.classif[k],l],imputedHd1[k,l])
      }
    }
    
    for (k in 1:569) {
      for (l in 1:11) {
        imputedBc1[k,l] = ifelse(is.na(imputedBc1[k,l]),as.data.frame(bc.som$codes)[bc.som$unit.classif[k],l],imputedBc1[k,l])
      }
    }
    for (k in 1:768) {
      for (l in 1:9) {
        imputedD1[k,l] = ifelse(is.na(imputedD1[k,l]),as.data.frame(d.som$codes)[d.som$unit.classif[k],l],imputedD1[k,l])
      }
    }
    
    imputedHd1 = as.data.frame(imputedHd1)
    imputedBc1 = as.data.frame(imputedBc1)
    imputedD1 = as.data.frame(imputedD1)
    
    #SOM weights are continuous,discrete variables must be converted back to the appropriate form.
    
    imputedHd1$num = round(imputedHd1$num)
    imputedHd1$fbs = round(imputedHd1$fbs)
    imputedHd1$cp = round(imputedHd1$cp)
    imputedHd1$sex = round(imputedHd1$sex)
    imputedHd1$exang = round(imputedHd1$exang)
    imputedHd1$slope = round(imputedHd1$slope)
    imputedHd1$thal = round(imputedHd1$thal)
    imputedHd1$restecg = round(imputedHd1$restecg)
    imputedHd1$ca = round(imputedHd1$ca)
    imputedBc1$diag = round(imputedBc1$diag)
    imputedD1$Outcome= round(imputedD1$Outcome)
    
    #Delete rows with class labels that were imputed.
    imputedHDel = imputedHd1[!is.na(hdMissing[,14]),]
    imputedBCDel = imputedBc1[!is.na(bcMissing[,1]),]
    imputedDDel = imputedD1[!is.na(dMissing[,9]),]
    
    hDel = hd[!is.na(hdMissing[,14]),]
    bcDel = bc[!is.na(bcMissing[,1]),]
    dDel = d[!is.na(dMissing[,9]),]
    
    #Specify training sets for the imputed data sets ~ 80% of the observations.
    trainSetH = sample(1:297,238)
    trainSetBC = sample(1:569,455)
    trainSetD = sample(1:768,614)
    trainSetHDel = sample(1:nrow(imputedHDel),round(nrow(imputedHDel)*0.8))
    trainSetBCDel = sample(1:nrow(imputedBCDel),round(nrow(imputedBCDel) * 0.8))
    trainSetDDel = sample(1:nrow(imputedDDel),round(nrow(imputedDDel) * 0.8))
    
    rows1 = (nrow(imputedHDel[trainSetHDel,]) - nrow(imputedHDel[-trainSetHDel,]))
    rows2 = (nrow(imputedBCDel[trainSetBCDel,]) - nrow(imputedBCDel[-trainSetBCDel,]))
    rows3 = (nrow(imputedDDel[trainSetDDel,]) - nrow(imputedDDel[-trainSetDDel,]))
    
    
    #Specify the training and test data for the GLM.
    xhSOM = model.matrix(num~.,imputedHd1[trainSetH,])[,-1]
    yhSOM = as.matrix(imputedHd1[trainSetH,14])
    xhSOMDel = model.matrix(num~.,imputedHDel[trainSetHDel,])[,-1]
    yhSOMDel = as.matrix(imputedHDel[trainSetHDel,14])
    testSetHSOM = rbind(as.matrix(imputedHd1[-trainSetH,]),matrix(0,nrow = 179,ncol = 14))
    testSetHSOMDel = rbind(as.matrix(imputedHDel[-trainSetHDel,]),matrix(0,nrow = rows1,ncol = 14))
    
    

    xbcSOM = model.matrix(diag~.,imputedBc1[trainSetBC,])[,-1]
    ybcSOM = as.matrix(imputedBc1[trainSetBC,1])
    xbcSOMDel = model.matrix(diag~.,imputedBCDel[trainSetBCDel,])[,-1]
    ybcSOMDel = as.matrix(imputedBCDel[trainSetBCDel,1])
    testSetBCSOM = rbind(as.matrix(imputedBc1[-trainSetBC,]),matrix(0,nrow = 341 ,ncol = 11))
    testSetBCSOMDel = rbind(as.matrix(imputedBCDel[-trainSetBCDel,]),matrix(0,nrow = rows2,ncol = 11))
    
    xdSOM = model.matrix(Outcome~.,imputedD1[trainSetD,])[,-1]
    ydSOM = as.matrix(imputedD1[trainSetD,9])
    xdSOMDel = model.matrix(Outcome~.,imputedDDel[trainSetDDel,])[,-1]
    ydSOMDel = as.matrix(imputedDDel[trainSetDDel,9])
    testSetDSOM = rbind(as.matrix(imputedD1[-trainSetD,]),matrix(0,nrow = 460,ncol = 9))
    testSetDSOMDel = rbind(as.matrix(imputedDDel[-trainSetDDel,]),matrix(0,nrow = rows3,ncol = 9))
    
    #Train the random forest for the data sets with and without deletions
    rfHD = randomForest(imputedHd1[trainSetH,1:13],imputedHd1[trainSetH,14])
    rfBC = randomForest(imputedBc1[trainSetBC,2:11],imputedBc1[trainSetBC,1])
    rfD = randomForest(imputedD1[trainSetD,1:8],imputedD1[trainSetD,9])
    
    rfHDDel = randomForest(imputedHDel[trainSetHDel,1:13],imputedHDel[trainSetHDel,14])
    rfBCDel = randomForest(imputedBCDel[trainSetBCDel,2:11],imputedBCDel[trainSetBCDel,1])
    rfDDel = randomForest(imputedDDel[trainSetDDel,1:8],imputedDDel[trainSetDDel,9])
    
    
    #LDA Training
    hdLDA = lda(as.formula("num ~ age + sex + cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal"),imputedHd1[trainSetH,])
    bcLDA = lda(as.formula("diag ~ radius + texture + perimeter + area + smoothness + compactness + concavity + concavepts + symmetry + fracdim"),imputedBc1[trainSetBC,])
    dLDA = lda(as.formula("Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + BMI + DiabetesPedigreeFunction + Age"),imputedD1[trainSetD,])
    
    hdLDADel = lda(as.formula("num ~ age + sex + cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal"),imputedHDel[trainSetHDel,])
    bcLDADel = lda(as.formula("diag ~ radius + texture + perimeter + area + smoothness + compactness + concavity + concavepts + symmetry + fracdim"),imputedBCDel[trainSetBCDel,])
    dLDADel = lda(as.formula("Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + BMI + DiabetesPedigreeFunction + Age"),imputedDDel[trainSetDDel,])
    
    #GLM Training
    hd.glm.SOM = cv.glmnet(xhSOM, yhSOM, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    hd.glm.SOM = glmnet(xhSOM, yhSOM, family = "binomial", alpha = 1, lambda = hd.glm.SOM$lambda.1se)
    hd.SOM.response = predict(hd.glm.SOM,newx = testSetHSOM[,-14],s = "lambda.min",type = "class")
    
    hd.glm.SOMDel = cv.glmnet(xhSOMDel, yhSOMDel, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    hd.glm.SOMDel = glmnet(xhSOMDel, yhSOMDel, family = "binomial", alpha = 1, lambda = hd.glm.SOMDel$lambda.1se)
    hd.SOM.responseDel = predict(hd.glm.SOMDel,newx = testSetHSOMDel[,-14],s = "lambda.min",type = "class")
    
    bc.glm.SOM = cv.glmnet(xbcSOM, ybcSOM, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    bc.glm.SOM = glmnet(xbcSOM, ybcSOM, family = "binomial", alpha = 1, lambda = bc.glm.SOM$lambda.1se)
    bc.SOM.response = predict(bc.glm.SOM,newx = testSetBCSOM[,-1],s = "lambda.min",type = "class")
    
    bc.glm.SOMDel = cv.glmnet(xbcSOMDel, ybcSOMDel, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    bc.glm.SOMDel = glmnet(xbcSOMDel, ybcSOMDel, family = "binomial", alpha = 1, lambda = bc.glm.SOMDel$lambda.1se)
    bc.SOM.responseDel = predict(bc.glm.SOMDel,newx = testSetBCSOMDel[,-1],s = "lambda.min",type = "class")
    
    d.glm.SOM = cv.glmnet(xdSOM, ydSOM, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    d.glm.SOM = glmnet(xdSOM, ydSOM, family = "binomial", alpha = 1, lambda = d.glm.SOM$lambda.1se)
    d.SOM.response = predict(d.glm.SOM,newx = testSetDSOM[,-9],s = "lambda.min",type = "class")
    
    d.glm.SOMDel = cv.glmnet(xdSOMDel, ydSOMDel, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    d.glm.SOMDel = glmnet(xdSOMDel, ydSOMDel, family = "binomial", alpha = 1, lambda = d.glm.SOMDel$lambda.1se)
    d.SOM.responseDel = predict(d.glm.SOMDel,newx = testSetDSOMDel[,-9],s = "lambda.min",type = "class")
    
  
    
    errorSOMGLM[1,j] = errorSOMGLM[1,j] + (1 - auc(as.numeric(hd[-trainSetH,14]),as.numeric(hd.SOM.response)[1:59]))
    errorSOMGLM[2,j] = errorSOMGLM[2,j] + (1 - auc(as.numeric(bc[-trainSetBC,1]),as.numeric(bc.SOM.response)[1:114]))
    errorSOMGLM[3,j] = errorSOMGLM[3,j] + (1 - auc(as.numeric(d[-trainSetD,9]),as.numeric(d.SOM.response)[1:154]))
    
    errorSOMGLMDel[1,j] = errorSOMGLMDel[1,j] + (1 - auc(as.numeric(hDel[-trainSetHDel,14]),as.numeric(hd.SOM.responseDel)[1:nrow(imputedHDel[-trainSetHDel,])]))
    errorSOMGLMDel[2,j] = errorSOMGLMDel[2,j] + (1 - auc(as.numeric(bcDel[-trainSetBCDel,1]),as.numeric(bc.SOM.responseDel)[1:nrow(imputedBCDel[-trainSetBCDel,])]))
    errorSOMGLMDel[3,j] = errorSOMGLMDel[3,j] + (1 - auc(as.numeric(dDel[-trainSetDDel,9]),as.numeric(d.SOM.responseDel)[1:nrow(imputedDDel[-trainSetDDel,])]))
    
    
    
    errorSOMLDA[1,j] = errorSOMLDA[1,j] + (1 - auc(as.numeric(hd[-trainSetH,14]),predict(hdLDA,imputedHd1[-trainSetH,])$class))
    errorSOMLDA[2,j] = errorSOMLDA[2,j] + (1 - auc(as.numeric(bc[-trainSetBC,1]),predict(bcLDA,imputedBc1[-trainSetBC,])$class))
    errorSOMLDA[3,j] = errorSOMLDA[3,j] + (1 - auc(as.numeric(d[-trainSetD,9]),predict(dLDA,imputedD1[-trainSetD,])$class))
    
    errorSOMLDADel[1,j] = errorSOMLDADel[1,j] + (1 - auc(as.numeric(hDel[-trainSetHDel,14]),predict(hdLDADel,imputedHDel[-trainSetHDel,])$class))
    errorSOMLDADel[2,j] = errorSOMLDADel[2,j] + (1 - auc(as.numeric(bcDel[-trainSetBCDel,1]),predict(bcLDADel,imputedBCDel[-trainSetBCDel,])$class))
    errorSOMLDADel[3,j] = errorSOMLDADel[3,j] + (1 - auc(as.numeric(dDel[-trainSetDDel,9]),predict(dLDADel,imputedDDel[-trainSetDDel,])$class))
    
    errorSOMRF[1,j] = errorSOMRF[1,j] + (1 - auc(as.numeric(hd[-trainSetH,14]),predict(rfHD,imputedHd1[-trainSetH,])))
    errorSOMRF[2,j] = errorSOMRF[2,j] + (1 - auc(as.numeric(bc[-trainSetBC,1]),predict(rfBC,imputedBc1[-trainSetBC,])))
    errorSOMRF[3,j] = errorSOMRF[3,j] + (1 - auc(as.numeric(d[-trainSetD,9]),predict(rfD,imputedD1[-trainSetD,])))
    
    
    errorSOMRFDel[1,j] = errorSOMRFDel[1,j] + (1 - auc(as.numeric(hDel[-trainSetHDel,14]),predict(rfHDDel,imputedHDel[-trainSetHDel,])))
    errorSOMRFDel[2,j] = errorSOMRFDel[2,j] + (1 - auc(as.numeric(bcDel[-trainSetBCDel,1]),predict(rfBCDel,imputedBCDel[-trainSetBCDel,])))
    errorSOMRFDel[3,j] = errorSOMRFDel[3,j] + (1 - auc(as.numeric(dDel[-trainSetDDel,9]),predict(rfDDel,imputedDDel[-trainSetDDel,])))

    
  }
}
#Average the SCE over all (n = 100) trials.

errorSOMGLM = errorSOMGLM / 100
errorSOMGLMDel = errorSOMGLMDel / 100
errorSOMLDA = errorSOMLDA / 100
errorSOMLDADel = errorSOMLDADel / 100
errorSOMRF = errorSOMRF / 100
errorSOMRFDel = errorSOMRFDel / 100

save(errorSOMGLM,file = "errorSOMGLM.Rdata")
save(errorSOMGLMDel,file = "errorSOMGLMDel.Rdata")
save(errorSOMLDA,file = "errorSOMLDA.Rdata")
save(errorSOMLDADel,file = "errorSOMLDADel.Rdata")
save(errorSOMRF,file = "errorSOMRF.Rdata")
save(errorSOMRFDel,file = "errorSOMRFDel.Rdata")

close(pb)
