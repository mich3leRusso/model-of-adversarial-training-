function [train_adv,label_adv]=create_adversarial(train,label,num,lgraph)  
 b=categorical(label)';
 T = onehotencode(b,2);
 %rimuovo il layer finale della rete in modo da calcolare la loss attuale
 %dei pattern 
 lgraph = removeLayers(lgraph,lgraph.Layers(end).Name);
 %numero di permutazioni per ciascun pattern
 m=4;
 for i=1:num %numero di pattren nel dataset 
     IM=train(:,:,:,i);
     label_adv(i)=label(i);
      maxloss=-20000000;
      net=dlnetwork(lgraph);
    for j=1:m
        SS=imnoise(IM,'gaussian');
         SD= dlarray(single(SS),"SSC");
        % Forward data through network
        Y = forward(net,SD);
        lb=T(i,:)';
        % Calculate cross-entropy loss.
        loss =  crossentropy(Y,lb,'TargetCategories','independent');
        if j==1
           maxloss=loss;
           SM=SS;
        end
        if loss>maxloss
            maxloss=loss;
            SM=SS;
        end
    end
    %inserisco il pattern 
    train_adv(:,:,:,i)=SM;
end 
end