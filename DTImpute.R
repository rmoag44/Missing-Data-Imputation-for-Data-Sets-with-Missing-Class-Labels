require(CoImp)
require(pROC)
require(glmnet)
require(mice)
require(MASS)
require(randomForest)


hd = read.csv("heartDisease.csv")
bc = read.csv("wdbc.csv")
d = read.csv("diabetes.csv")

errorDTGLM = matrix(0,nrow = 3,ncol = 9)
errorDTLDA = matrix(0,nrow = 3,ncol = 9)
errorDTRF = matrix(0,nrow = 3,ncol = 9)
errorDTGLMDel = matrix(0,nrow = 3,ncol = 9)
errorDTLDADel = matrix(0,nrow = 3,ncol = 9)
errorDTRFDel = matrix(0,nrow = 3,ncol = 9)

pb <- txtProgressBar(min = 1, max = 100, style = 3)

for (i in 1:100) {
  set.seed(i)
  setTxtProgressBar(pb, i)
  for (j in 1:9) {
    
    hdMissing = MCAR(as.matrix(hd),perc.miss = 0.05 * j,setseed = i)@db.missing
    bcMissing = MCAR(as.matrix(bc),perc.miss = 0.05 * j,setseed = i)@db.missing
    dMissing = MCAR(as.matrix(d),perc.miss = 0.05 * j,setseed = i)@db.missing
    
    
    ptm = proc.time()
    imputedHd2 = complete(mice(hdMissing,method = "cart",printFlag = FALSE))
    execTimeDT[1,j] = execTimeDT[1,j] + (proc.time() - ptm)[3]
    
    ptm = proc.time()
    imputedBc2 = complete(mice(bcMissing,method = "cart",printFlag = FALSE))
    execTimeDT[2,j] = execTimeDT[2,j] + (proc.time() - ptm)[3]
    
    ptm = proc.time()
    imputedD2 = complete(mice(dMissing,method = "cart",printFlag = FALSE))
    execTimeDT[3,j] = execTimeDT[3,j] + (proc.time() - ptm)[3]
    
    imputedHDel = imputedHd2[!is.na(hdMissing[,14]),]
    imputedBCDel = imputedBc2[!is.na(bcMissing[,1]),]
    imputedDDel = imputedD2[!is.na(dMissing[,9]),]
    
    hDel = hd[!is.na(hdMissing[,14]),]
    bcDel = bc[!is.na(bcMissing[,1]),]
    dDel = d[!is.na(dMissing[,9]),]
    
    
    trainSetH = sample(1:297,238)
    trainSetBC = sample(1:569,455)
    trainSetD = sample(1:768,614)
    trainSetHDel = sample(1:nrow(imputedHDel),round(nrow(imputedHDel)*0.8))
    trainSetBCDel = sample(1:nrow(imputedBCDel),round(nrow(imputedBCDel) * 0.8))
    trainSetDDel = sample(1:nrow(imputedDDel),round(nrow(imputedDDel) * 0.8))
    
    rows1 = (nrow(imputedHDel[trainSetHDel,]) - nrow(imputedHDel[-trainSetHDel,]))
    rows2 = (nrow(imputedBCDel[trainSetBCDel,]) - nrow(imputedBCDel[-trainSetBCDel,]))
    rows3 = (nrow(imputedDDel[trainSetDDel,]) - nrow(imputedDDel[-trainSetDDel,]))
    
    rfHD = randomForest(imputedHd2[trainSetH,1:13],imputedHd2[trainSetH,14])
    rfBC = randomForest(imputedBc2[trainSetBC,2:11],imputedBc2[trainSetBC,1])
    rfD = randomForest(imputedD2[trainSetD,1:8],imputedD2[trainSetD,9])
    
    rfHDDel = randomForest(imputedHDel[trainSetHDel,1:13],imputedHDel[trainSetHDel,14])
    rfBCDel = randomForest(imputedBCDel[trainSetBCDel,2:11],imputedBCDel[trainSetBCDel,1])
    rfDDel = randomForest(imputedDDel[trainSetDDel,1:8],imputedDDel[trainSetDDel,9])
    
    hdLDA = lda(as.formula("num ~ age + sex + cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal"),imputedHd2[trainSetH,])
    bcLDA = lda(as.formula("diag ~ radius + texture + perimeter + area + smoothness + compactness + concavity + concavepts + symmetry + fracdim"),imputedBc2[trainSetBC,])
    dLDA = lda(as.formula("Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + BMI + DiabetesPedigreeFunction + Age"),imputedD2[trainSetD,])
    
    hdLDADel = lda(as.formula("num ~ age + sex + cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal"),imputedHDel[trainSetHDel,])
    bcLDADel = lda(as.formula("diag ~ radius + texture + perimeter + area + smoothness + compactness + concavity + concavepts + symmetry + fracdim"),imputedBCDel[trainSetBCDel,])
    dLDADel = lda(as.formula("Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + BMI + DiabetesPedigreeFunction + Age"),imputedDDel[trainSetDDel,])
    

    
    xhDT = model.matrix(num~.,imputedHd2[trainSetH,])[,-1]
    yhDT = as.matrix(imputedHd2[trainSetH,14])
    testSetDT = rbind(as.matrix(imputedHd2[-trainSetH,]),matrix(0,nrow = 179,ncol = 14))
    xhDTDel = model.matrix(num~.,imputedHDel[trainSetHDel,])[,-1]
    yhDTDel = as.matrix(imputedHDel[trainSetHDel,14])
    testSetHDTDel = rbind(as.matrix(imputedHDel[-trainSetHDel,]),matrix(0,nrow = rows1,ncol = 14))
    
    xbcDT = model.matrix(diag~.,imputedBc2[trainSetBC,])[,-1]
    ybcDT = as.matrix(imputedBc2[trainSetBC,1])
    testSetBCDT = rbind(as.matrix(imputedBc2[-trainSetBC,]),matrix(0,nrow = 341 ,ncol = 11))
    xbcDTDel = model.matrix(diag~.,imputedBCDel[trainSetBCDel,])[,-1]
    ybcDTDel = as.matrix(imputedBCDel[trainSetBCDel,1])
    testSetBCDTDel = rbind(as.matrix(imputedBCDel[-trainSetBCDel,]),matrix(0,nrow = rows2,ncol = 11))
    
    xdDT = model.matrix(Outcome~.,imputedD2[trainSetD,])[,-1]
    ydDT = as.matrix(imputedD2[trainSetD,9])
    testSetDDT = rbind(as.matrix(imputedD2[-trainSetD,]),matrix(0,nrow = 460,ncol = 9))
    xdDTDel = model.matrix(Outcome~.,imputedDDel[trainSetDDel,])[,-1]
    ydDTDel = as.matrix(imputedDDel[trainSetDDel,9])
    testSetDDTDel = rbind(as.matrix(imputedDDel[-trainSetDDel,]),matrix(0,nrow = rows3,ncol = 9))
    
    hd.glm.DT = cv.glmnet(xhDT, yhDT, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    hd.glm.DT = glmnet(xhDT, yhDT, family = "binomial", alpha = 1, lambda = hd.glm.DT$lambda.1se)
    hd.DT.response = predict(hd.glm.DT,newx = testSetDT[,-14],s = "lambda.min",type = "class")
    
    hd.glm.DTDel = cv.glmnet(xhDTDel, yhDTDel, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    hd.glm.DTDel = glmnet(xhDTDel, yhDTDel, family = "binomial", alpha = 1, lambda = hd.glm.DTDel$lambda.1se)
    hd.DT.responseDel = predict(hd.glm.DTDel,newx = testSetHDTDel[,-14],s = "lambda.min",type = "class")
    
    
    bc.glm.DT = cv.glmnet(xbcDT, ybcDT, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    bc.glm.DT = glmnet(xbcDT, ybcDT, family = "binomial", alpha = 1, lambda = bc.glm.DT$lambda.1se)
    bc.DT.response = predict(bc.glm.DT,newx = testSetBCDT[,-1],s = "lambda.min",type = "class")
    
    bc.glm.DTDel = cv.glmnet(xbcDTDel, ybcDTDel, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    bc.glm.DTDel = glmnet(xbcDTDel, ybcDTDel, family = "binomial", alpha = 1, lambda = bc.glm.DTDel$lambda.1se)
    bc.DT.responseDel = predict(bc.glm.DTDel,newx = testSetBCDTDel[,-1],s = "lambda.min",type = "class")
    
    
    
    d.glm.DT = cv.glmnet(xdDT, ydDT, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    d.glm.DT = glmnet(xdDT, ydDT, family = "binomial", alpha = 1, lambda = d.glm.DT$lambda.1se)
    d.DT.response = predict(d.glm.DT,newx = testSetDDT[,-9],s = "lambda.min",type = "class")
    
    d.glm.DTDel = cv.glmnet(xdDTDel, ydDTDel, family = "binomial", alpha = 1, type.measure = "class",nlambda = 100)
    d.glm.DTDel = glmnet(xdDTDel, ydDTDel, family = "binomial", alpha = 1, lambda = d.glm.DTDel$lambda.1se)
    d.DT.responseDel = predict(d.glm.DTDel,newx = testSetDDTDel[,-9],s = "lambda.min",type = "class")
    
    
    errorDTGLM[1,j] = errorDTGLM[1,j] + (1 - auc(as.numeric(as.matrix(hd[-trainSetH,14])),as.numeric(hd.DT.response)[1:59]))
    errorDTGLM[2,j] = errorDTGLM[2,j] + (1 - auc(as.numeric(as.matrix(bc[-trainSetBC,1])),as.numeric(bc.DT.response)[1:114]))
    errorDTGLM[3,j] = errorDTGLM[3,j] + (1 - auc(as.numeric(as.matrix(d[-trainSetD,9])),as.numeric(d.DT.response)[1:154]))
    
    errorDTGLMDel[1,j] = errorDTGLMDel[1,j] + (1 - auc(as.numeric(as.matrix(hDel[-trainSetHDel,14])),as.numeric(hd.DT.responseDel)[1:nrow(imputedHDel[-trainSetHDel,])]))
    errorDTGLMDel[2,j] = errorDTGLMDel[2,j] + (1 - auc(as.numeric(as.matrix(bcDel[-trainSetBCDel,1])),as.numeric(bc.DT.responseDel)[1:nrow(imputedBCDel[-trainSetBCDel,])]))
    errorDTGLMDel[3,j] = errorDTGLMDel[3,j] + (1 - auc(as.numeric(as.matrix(dDel[-trainSetDDel,9])),as.numeric(d.DT.responseDel)[1:nrow(imputedDDel[-trainSetDDel,])]))
    
    errorDTLDA[1,j] = errorDTLDA[1,j] + (1 - auc(as.numeric(as.matrix(hd[-trainSetH,14])),predict(hdLDA,imputedHd2[-trainSetH,])$class))
    errorDTLDA[2,j] = errorDTLDA[2,j] + (1 - auc(as.numeric(as.matrix(bc[-trainSetBC,1])),predict(bcLDA,imputedBc2[-trainSetBC,])$class))
    errorDTLDA[3,j] = errorDTLDA[3,j] + (1 - auc(as.numeric(as.matrix(d[-trainSetD,9])),predict(dLDA,imputedD2[-trainSetD,])$class))
    
    errorDTLDADel[1,j] = errorDTLDADel[1,j] + (1 - auc(as.numeric(as.matrix(hDel[-trainSetHDel,14])),predict(hdLDADel,imputedHDel[-trainSetHDel,])$class))
    errorDTLDADel[2,j] = errorDTLDADel[2,j] + (1 - auc(as.numeric(as.matrix(bcDel[-trainSetBCDel,1])),predict(bcLDADel,imputedBCDel[-trainSetBCDel,])$class))
    errorDTLDADel[3,j] = errorDTLDADel[3,j] + (1 - auc(as.numeric(as.matrix(dDel[-trainSetDDel,9])),predict(dLDADel,imputedDDel[-trainSetDDel,])$class))
    
    errorDTRF[1,j] = errorDTRF[1,j] + (1 - auc(as.numeric(as.matrix(hd[-trainSetH,14])),predict(rfHD,imputedHd2[-trainSetH,])))
    errorDTRF[2,j] = errorDTRF[2,j] + (1 - auc(as.numeric(as.matrix(bc[-trainSetBC,1])),predict(rfBC,imputedBc2[-trainSetBC,])))
    errorDTRF[3,j] = errorDTRF[3,j] + (1 - auc(as.numeric(as.matrix(d[-trainSetD,9])),predict(rfD,imputedD2[-trainSetD,])))
    
    errorDTRFDel[1,j] = errorDTRFDel[1,j] + (1 - auc(as.numeric(as.matrix(hDel[-trainSetHDel,14])),predict(rfHDDel,imputedHDel[-trainSetHDel,])))
    errorDTRFDel[2,j] = errorDTRFDel[2,j] + (1 - auc(as.numeric(as.matrix(bcDel[-trainSetBCDel,1])),predict(rfBCDel,imputedBCDel[-trainSetBCDel,])))
    errorDTRFDel[3,j] = errorDTRFDel[3,j] + (1 - auc(as.numeric(as.matrix(dDel[-trainSetDDel,9])),predict(rfDDel,imputedDDel[-trainSetDDel,])))
  }
}
errorDTGLM = errorDTGLM / 100
errorDTGLMDel = errorDTGLMDel / 100
errorDTLDA = errorDTLDA / 100
errorDTLDADel = errorDTLDADel / 100
errorDTRF = errorDTRF / 100
errorDTRFDel = errorDTRFDel / 100

save(errorDTGLM,file = "errorDTGLM.Rdata")
save(errorDTGLMDel,file = "errorDTGLMDel.Rdata")
save(errorDTLDA,file = "errorDTLDA.Rdata")
save(errorDTLDADel,file = "errorDTLDADel.Rdata")
save(errorDTRF,file = "errorDTRF.Rdata")
save(errorDTRFDel,file = "errorDTRFDel.Rdata")

close(pb)