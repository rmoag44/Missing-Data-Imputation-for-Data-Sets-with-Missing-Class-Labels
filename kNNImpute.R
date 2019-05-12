require(CoImp)
require(VIM)
require(pROC)
require(glmnet)
require(mice)
require(MASS)
require(randomForest)

hd = read.csv("heartDisease.csv")
bc = read.csv("wdbc.csv")
d = read.csv("diabetes.csv")

errorkNNGLM = matrix(0,nrow = 3,ncol = 9)
errorkNNLDA = matrix(0,nrow = 3,ncol = 9)
errorkNNRF = matrix(0,nrow = 3,ncol = 9)
errorkNNGLMDel = matrix(0,nrow = 3,ncol = 9)
errorkNNLDADel = matrix(0,nrow = 3,ncol = 9)
errorkNNRFDel = matrix(0,nrow = 3,ncol = 9)

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
    
    # Perform 5-NN Imputation, kNN returns the imputed data set as a list concatenated with a TRUE/FALSE matrix for each
    # variable indicating whether or not it was imputed. Subset the imputed data.
    
    imputedHd1 = kNN(hdMissing)
    imputedHd1 = subset(imputedHd1,select = age:num)
    
    imputedBc1 = kNN(bcMissing)
    imputedBc1 = subset(imputedBc1,select = diag:fracdim)
    
    imputedD1 = kNN(dMissing)
    imputedD1 = subset(imputedD1,select = Pregnancies:Outcome)
    
    #List must be coerced into a data frame.Functions that will be used later only accept matrices or data frames as arguments.
    
    imputedHd1 = as.data.frame(imputedHd1)
    imputedBc1 = as.data.frame(imputedBc1)
    imputedD1 = as.data.frame(imputedD1)
    
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
    
    xhKNN = model.matrix(num~.,imputedHd1[trainSetH,])[,-1]
    yhKNN = as.matrix(imputedHd1[trainSetH,14])
    testSetHKNN = rbind(as.matrix(imputedHd1[-trainSetH,]),matrix(0,nrow = 179,ncol = 14))
    xhKNNDel = model.matrix(num~.,imputedHDel[trainSetHDel,])[,-1]
    yhKNNDel = as.matrix(imputedHDel[trainSetHDel,14])
    testSetHKNNDel = rbind(as.matrix(imputedHDel[-trainSetHDel,]),matrix(0,nrow = rows1,ncol = 14))
    
    xbcKNN = model.matrix(diag~.,imputedBc1[trainSetBC,])[,-1]
    ybcKNN = as.matrix(imputedBc1[trainSetBC,1])
    xbcKNNDel = model.matrix(diag~.,imputedBCDel[trainSetBCDel,])[,-1]
    ybcKNNDel = as.matrix(imputedBCDel[trainSetBCDel,1])
    testSetBCKNN = rbind(as.matrix(imputedBc1[-trainSetBC,]),matrix(0,nrow = 341 ,ncol = 11))
    testSetBCKNNDel = rbind(as.matrix(imputedBCDel[-trainSetBCDel,]),matrix(0,nrow = rows2,ncol = 11))
    
    xdKNN = model.matrix(Outcome~.,imputedD1[trainSetD,])[,-1]
    ydKNN = as.matrix(imputedD1[trainSetD,9])
    xdKNNDel = model.matrix(Outcome~.,imputedDDel[trainSetDDel,])[,-1]
    ydKNNDel = as.matrix(imputedDDel[trainSetDDel,9])
    testSetDKNN = rbind(as.matrix(imputedD1[-trainSetD,]),matrix(0,nrow = 460,ncol = 9))
    testSetDKNNDel = rbind(as.matrix(imputedDDel[-trainSetDDel,]),matrix(0,nrow = rows3,ncol = 9))
    
    #Train the random forest for the data sets with and without deletions.

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
    
    hd.glm.kNN = cv.glmnet(xhKNN, yhKNN, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    hd.glm.kNN = glmnet(xhKNN, yhKNN, family = "binomial", alpha = 1, lambda = hd.glm.kNN$lambda.1se)
    hd.kNN.response = predict(hd.glm.kNN,newx = testSetHKNN[,-14],s = "lambda.min",type = "class")
    
    hd.glm.kNNDel = cv.glmnet(xhKNNDel, yhKNNDel, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    hd.glm.kNNDel = glmnet(xhKNNDel, yhKNNDel, family = "binomial", alpha = 1, lambda = hd.glm.kNNDel$lambda.1se)
    hd.kNN.responseDel = predict(hd.glm.kNNDel,newx = testSetHKNNDel[,-14],s = "lambda.min",type = "class")
    
    bc.glm.kNN = cv.glmnet(xbcKNN, ybcKNN, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    bc.glm.kNN = glmnet(xbcKNN, ybcKNN, family = "binomial", alpha = 1, lambda = bc.glm.kNN$lambda.1se)
    bc.kNN.response = predict(bc.glm.kNN,newx = testSetBCKNN[,-1],s = "lambda.min",type = "class")
    
    bc.glm.kNNDel = cv.glmnet(xbcKNNDel, ybcKNNDel, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    bc.glm.kNNDel = glmnet(xbcKNNDel, ybcKNNDel, family = "binomial", alpha = 1, lambda = bc.glm.kNNDel$lambda.1se)
    bc.kNN.responseDel = predict(bc.glm.kNNDel,newx = testSetBCKNNDel[,-1],s = "lambda.min",type = "class")
    
    d.glm.kNN = cv.glmnet(xdKNN, ydKNN, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    d.glm.kNN = glmnet(xdKNN, ydKNN, family = "binomial", alpha = 1, lambda = d.glm.kNN$lambda.1se)
    d.kNN.response = predict(d.glm.kNN,newx = testSetDKNN[,-9],s = "lambda.min",type = "class")
    
    d.glm.kNNDel = cv.glmnet(xdKNNDel, ydKNNDel, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    d.glm.kNNDel = glmnet(xdKNNDel, ydKNNDel, family = "binomial", alpha = 1, lambda = d.glm.kNNDel$lambda.1se)
    d.kNN.responseDel = predict(d.glm.kNNDel,newx = testSetDKNNDel[,-9],s = "lambda.min",type = "class")
    
    #Calculate the SCE for each model on each data set.
    
    errorkNNGLM[1,j] = errorkNNGLM[1,j] + (1 - auc(as.numeric(hd[-trainSetH,14]),as.numeric(hd.kNN.response)[1:59]))
    errorkNNGLM[2,j] = errorkNNGLM[2,j] + (1 - auc(as.numeric(bc[-trainSetBC,1]),as.numeric(bc.kNN.response)[1:114]))
    errorkNNGLM[3,j] = errorkNNGLM[3,j] + (1 - auc(as.numeric(d[-trainSetD,9]),as.numeric(d.kNN.response)[1:154]))
    
    errorkNNGLMDel[1,j] = errorkNNGLMDel[1,j] + (1 - auc(as.numeric(hDel[-trainSetHDel,14]),as.numeric(hd.kNN.responseDel)[1:nrow(imputedHDel[-trainSetHDel,])]))
    errorkNNGLMDel[2,j] = errorkNNGLMDel[2,j] + (1 - auc(as.numeric(bcDel[-trainSetBCDel,1]),as.numeric(bc.kNN.responseDel)[1:nrow(imputedBCDel[-trainSetBCDel,])]))
    errorkNNGLMDel[3,j] = errorkNNGLMDel[3,j] + (1 - auc(as.numeric(dDel[-trainSetDDel,9]),as.numeric(d.kNN.responseDel)[1:nrow(imputedDDel[-trainSetDDel,])]))
  
     
    errorkNNLDA[1,j] = errorkNNLDA[1,j] + (1 - auc(as.numeric(hd[-trainSetH,14]),predict(hdLDA,imputedHd1[-trainSetH,])$class))
    errorkNNLDA[2,j] = errorkNNLDA[2,j] + (1 - auc(as.numeric(bc[-trainSetBC,1]),predict(bcLDA,imputedBc1[-trainSetBC,])$class))
    errorkNNLDA[3,j] = errorkNNLDA[3,j] + (1 - auc(as.numeric(d[-trainSetD,9]),predict(dLDA,imputedD1[-trainSetD,])$class))
     
    errorkNNLDADel[1,j] = errorkNNLDADel[1,j] + (1 - auc(as.numeric(hDel[-trainSetHDel,14]),predict(hdLDADel,imputedHDel[-trainSetHDel,])$class))
    errorkNNLDADel[2,j] = errorkNNLDADel[2,j] + (1 - auc(as.numeric(bcDel[-trainSetBCDel,1]),predict(bcLDADel,imputedBCDel[-trainSetBCDel,])$class))
    errorkNNLDADel[3,j] = errorkNNLDADel[3,j] + (1 - auc(as.numeric(dDel[-trainSetDDel,9]),predict(dLDADel,imputedDDel[-trainSetDDel,])$class))
 
    errorkNNRF[1,j] = errorkNNRF[1,j] + (1 - auc(as.numeric(hd[-trainSetH,14]),predict(rfHD,imputedHd1[-trainSetH,])))
    errorkNNRF[2,j] = errorkNNRF[2,j] + (1 - auc(as.numeric(bc[-trainSetBC,1]),predict(rfBC,imputedBc1[-trainSetBC,])))
    errorkNNRF[3,j] = errorkNNRF[3,j] + (1 - auc(as.numeric(d[-trainSetD,9]),predict(rfD,imputedD1[-trainSetD,])))
    
    errorkNNRFDel[1,j] = errorkNNRFDel[1,j] + (1 - auc(as.numeric(as.matrix(hDel[-trainSetHDel,14])),predict(rfHDDel,imputedHDel[-trainSetHDel,])))
    errorkNNRFDel[2,j] = errorkNNRFDel[2,j] + (1 - auc(as.numeric(as.matrix(bcDel[-trainSetBCDel,1])),predict(rfBCDel,imputedBCDel[-trainSetBCDel,])))
    errorkNNRFDel[3,j] = errorkNNRFDel[3,j] + (1 - auc(as.numeric(as.matrix(dDel[-trainSetDDel,9])),predict(rfDDel,imputedDDel[-trainSetDDel,])))
  }

}

#Average the SCE over all (n = 100) trials.

errorkNNGLM = errorkNNGLM / 100
errorkNNGLMDel = errorkNNGLMDel / 100
errorkNNLDA = errorkNNLDA / 100
errorkNNLDADel = errorkNNLDADel / 100
errorkNNRF = errorkNNRF / 100
errorkNNRFDel = errorkNNRFDel / 100

save(errorkNNGLM,file = "errorkNNGLM.Rdata")
save(errorkNNGLMDel,file = "errorkNNGLMDel.Rdata")
save(errorkNNLDA,file = "errorkNNLDA.Rdata")
save(errorkNNLDADel,file = "errorkNNLDADel.Rdata")
save(errorkNNRF,file = "errorkNNRF.Rdata")
save(errorkNNRFDel,file = "errorkNNRFDel.Rdata")

close(pb)