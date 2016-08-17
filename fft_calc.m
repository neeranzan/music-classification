clear;
%classical 1
%country  2
%jazz     3
%pop      4
%rock     5
%metal    6

label=[];
features=1000;
instances=600;
class=6;


sd=[];


%import data
for i=0:9
    
    
    filepath=strcat('opihi.cs.uvic.ca/sound/genres/country/country.0000',num2str(i),'.wav');
    x=audioread(filepath);
    x=fft(x(1:1000))';
    sd=[sd;x];
    label=[label;2];
    
    
    filepath=strcat('opihi.cs.uvic.ca/sound/genres/pop/pop.0000',num2str(i),'.wav');
    x=audioread(filepath);
    x=fft(x(1:1000))';
    sd=[sd;x];
    label=[label;4];
    
    filepath=strcat('opihi.cs.uvic.ca/sound/genres/classical/classical.0000',num2str(i),'.wav');
    x=audioread(filepath);
    x=fft(x(1:1000))';
    sd=[sd;x];
    label=[label;1];
    
    
    
    filepath=strcat('opihi.cs.uvic.ca/sound/genres/jazz/jazz.0000',num2str(i),'.wav');
    x=audioread(filepath);
    x=fft(x(1:1000))';
    sd=[sd;x];
    label=[label;3];
    
    
    
    filepath=strcat('opihi.cs.uvic.ca/sound/genres/rock/rock.0000',num2str(i),'.wav');
    x=audioread(filepath);
    x=fft(x(1:1000))';
    sd=[sd;x];
    label=[label;5];
    
    
    filepath=strcat('opihi.cs.uvic.ca/sound/genres/metal/metal.0000',num2str(i),'.wav');
    x=audioread(filepath);
    x=fft(x(1:1000))';
    sd=[sd;x];
    label=[label;6];
    
end


for i=10:99
    
    filepath=strcat('opihi.cs.uvic.ca/sound/genres/jazz/jazz.000',num2str(i),'.wav');
    x=audioread(filepath);
    x=fft(x(1:1000))';
    sd=[sd;x];
    label=[label;3];
    
    
    
    filepath=strcat('opihi.cs.uvic.ca/sound/genres/country/country.000',num2str(i),'.wav');
    x=audioread(filepath);
    x=fft(x(1:1000))';
    sd=[sd;x];
    label=[label;2];
    
    
    filepath=strcat('opihi.cs.uvic.ca/sound/genres/metal/metal.000',num2str(i),'.wav');
    x=audioread(filepath);
    x=fft(x(1:1000))';
    sd=[sd;x];
    label=[label;6];
    
    filepath=strcat('opihi.cs.uvic.ca/sound/genres/classical/classical.000',num2str(i),'.wav');
    x=audioread(filepath);
    x=fft(x(1:1000))';
    sd=[sd;x];
    label=[label;1];
    
    
    
    filepath=strcat('opihi.cs.uvic.ca/sound/genres/pop/pop.000',num2str(i),'.wav');
    x=audioread(filepath);
    x=fft(x(1:1000))';
    sd=[sd;x];
    label=[label;4];
    
    filepath=strcat('opihi.cs.uvic.ca/sound/genres/rock/rock.000',num2str(i),'.wav');
    x=audioread(filepath);
    x=fft(x(1:1000))';
    sd=[sd;x];
    label=[label;5];
    
    
    
    
end

clear x;
clear i;
clear filepath;

%get maximum value for each column
%MaxElem=max(sd);

%normalize the matrix
%for i=1:feat
%sd(:,i)=sd(:,i)/MaxElem(i);
%end;

fold=10;
trainlabel=[];
testlabel=[];

step_size=0.01; %learning rate
lambda=0.001;  %1 or 10
epoch=200;
wo=0;

%confusion matrix
cm=zeros(class,class);

for f= 0: fold-1 %use fold later
    
    % eval('f');
    
    train=[];
    test=[];
    trainlabel=[];
    testlabel=[];
    
    
    accuracy =0;
    %weights
    w_prev=zeros(6,1000);
    w=zeros(6,1000);% make w0=1 for all cases
    
    
    
    
    for k= 1:instances
        
        if(mod(k,fold)==f)
            % eval('f');
            
            test=[test;sd(k,:)];
            testlabel=[testlabel;label(k)];
        else
            train=[train;sd(k,:)];
            trainlabel=[trainlabel;label(k)];
        end
    end
    
    [n,~]=size(trainlabel);
    [tn,~]=size(testlabel);
    
    % eval('n');
    %eval('tn');
    
    %normalization process
    trainmax=max(train);
    for i=1:540
        train(i,:)=train(i,:) ./ trainmax;
    end
    
    for i=1:60
        test(i,:)=test(i,:) ./trainmax;
    end
    
    %load delta matrix
    delta=zeros(6,540);
    
    for j=1: 6
        for l=1:540
            if(trainlabel(l)==j)
                delta(j,l)=1;
            else
                delta(j,l)=0;
            end
        end
    end
    
    for e=1:epoch
        
       % eval('e');
        prob=zeros(540,6); %probability for all classes for each instance of train
        probTest=zeros(60,6);%probability for all class for each instance of test
        
        
        %------calculate probabilities for each instance using weights
        
        %for each instance
        for l=1:540
            
            %delta=getDeltaValue(j,l);
            
            
            
            for pk=1:class
                
                %class summation
                c_s=0;
                
                numerator=sum(w(pk,:) .* train(l,:),2);
                %feature summation
                f_s=0;
                for k=1:class;
                    
                    % (weight * Xi) and summation of it
                    f_s=sum(w(k,:) .* train(l,:),2);
                    %class summation
                    c_s=c_s+ exp(wo+f_s);
                    
                end
                
                prob(l,pk)=exp(wo+numerator)/(1 + c_s);
                
            end
            %prob of remaining one
            
            if(l==539)
                test1=5;
            end
           % prob(l,class)=1-sum(prob(l,:),2);
            
        end
        
        
        [probMax,maxIndex]=max(prob,[],2);
        
        %calculate accuracy
        correct=0;
        for i=1:540
            if(maxIndex(i)==trainlabel(i))
                correct=correct+1;
            end
            
        end
        
        %eval('correct/540');
        
        if(accuracy<=correct/540)
            
            %continue gradient descent
            accuracy=correct/540;
            
            %continue changing weights
            %gradient descent
            
            w_prev=w;
            
            %gradient_summation
            g_s=zeros(6,1000);
            
            for j=1:class
                
                for i=1:features
                    
                    for l=1:540
                        
                        g_s(j,i)=g_s(j,i)+train(l,i)*(delta(j,l)-prob(l,j))- (step_size*lambda*w(j,i));
                        
                    end
                    
                    w(j,i)=w(j,i)+step_size *g_s(j,i);
                    
                end
                
                
            end
            
             %step_size=step_size/(1 +e/epoch);
            %  eval('step_size');
        else
            
            %calculate probabilities for testdata with new weights
            
            
            
            %for each test instance
            for l=1:60
                
                %delta=getDeltaValue(j,l);
                
                
                for pk=1:class
                    
                    %class summation
                    c_s=0;
                    
                    numerator=sum(w_prev(pk,:) .* test(l,:),2);
                    %feature summation
                    f_s=0;
                    for k=1:class;
                        
                        % (weight * Xi) and summation of it
                        f_s=sum(w_prev(k,:) .* test(l,:),2);
                        %class summation
                        c_s=c_s+ exp(wo+f_s);
                        
                    end
                    
                    probTest(l,pk)=exp(wo+numerator)/(1 + c_s);
                    
                end
                %prob of remaining one
              %  probTest(l,class)=1-sum(probTest(l,:),2);
                
            end
            
            %build/update confusion matrix
            [probTestMax,maxTestIndex]=max(probTest,[],2);
            for i=1:60
                cm(testlabel(i),maxTestIndex(i))=cm(testlabel(i),maxTestIndex(i))+1;
                
            end
            
            
            break
            
        end
        
        
    end
end
%calculate accuracy rate
accurate=0
for i=1:class
    for j=1:class
        if(i==j)
            accurate=accurate+cm(i,j);
        end
    end
end

accuracy=accurate/600;
eval('accuracy');


