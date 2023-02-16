n = 8;
RDM.dist28{n} = pdist(prompt_Flow_RoBERTa','cosine');
RDM.distM{n} = squareform(RDM.dist28{n});
RDM.dist27{n} = squareform(RDM.distM{n}(1:27,1:27));

%%
similarity = table;
n = 0;
for i = 1:height(RDM)
    for k = i+1:height(RDM)
        n = n+1;
        similarity.RDM1{n,1} = RDM.source{i};
        similarity.RDM2{n,1} = RDM.source{k};
        [similarity.r_28{n,1}, similarity.p_28{n,1}] = corr(RDM.dist28{i}',RDM.dist28{k}','type','spearman');
        [similarity.r_27{n,1}, similarity.p_27{n,1}] = corr(RDM.dist27{i}',RDM.dist27{k}','type','spearman');
    end
end
clear i k n;
%%
dist27 = array2table(cell2mat(RDM{:,5})','VariableNames',{'Subjective','Annotation','Embedding_GloVe','Embedding_BERT','Embedding_RoBERTa','Prompt_BERT','Prompt_RoBERTa'});
writetable(dist27,'dist_OnlyBehavior.xlsx','FileType','spreadsheet');

