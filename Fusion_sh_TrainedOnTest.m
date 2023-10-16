clc,clear all
%------------------------
load('FusionScores.mat')
load('y.mat')
Labels=y;
%------------------------


%------------------------
% normalization:
% scores_cosine=mapminmax(scores_cosine);
% scores_GuassianBackend=mapminmax(scores_GuassianBackend);
% scores_Lda150Gplda=mapminmax(scores_Lda150Gplda);
% scores_moplda=mapminmax(scores_moplda);

scores_cosine=mapstd(scores_cosine);
scores_GuassianBackend=mapstd(scores_GuassianBackend);
scores_Lda150Gplda=mapstd(scores_Lda150Gplda);
scores_moplda=mapstd(scores_moplda);
%------------------------


%------------------------
k=0; j=0;
for i=1:length(Labels)
    if Labels(i)==1
        k=k+1;
        TargetScores(1:4,k)=[scores_cosine(i),scores_GuassianBackend(i),scores_Lda150Gplda(i),scores_moplda(i)];
    else
        j=j+1;
        NonTargetScores(1:4,j)=[scores_cosine(i),scores_GuassianBackend(i),scores_Lda150Gplda(i),scores_moplda(i)];
    end
end
%------------------------


%------------------------
%Train fusion weights
%w0 = train_llr_fusion(TargetScores,NonTargetScores);
%w = train_mse_fusion(TargetScores,NonTargetScores,0.5,w0);
%w=train_mse_fusion(TargetScores,NonTargetScores);
w = train_llr_fusion(TargetScores,NonTargetScores);
%------------------------


%------------------------
%w=w0
TargetScores2=TargetScores;
NonTargetScores2=NonTargetScores;
TargetScores2(5,:)=1;
NonTargetScores2(5,:)=1;
TargetScores2=w'*TargetScores2;
NonTargetScores2=w'*NonTargetScores2;
%------------------------



%------------------------
k=0; j=0; Scores=[];
for i=1:length(Labels)
    if Labels(i)==1
        k=k+1;
        Scores(i)=TargetScores2(k);
    else
        j=j+1;
        Scores(i)=NonTargetScores2(j);
    end
end
%------------------------

ypred=Scores;
save('ypred','ypred');


