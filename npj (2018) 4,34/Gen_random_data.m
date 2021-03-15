function result = Gen_random_data(sample_num)
%result is 3 dimensional matrix
result = zeros(sample_num,15);
label = zeros(sample_num,1);
for i = 1:sample_num
    [result(i,:),label(i)] = matrix2vector(RandomDensityMatrix(4));
end
save('mat/result','result');
save('mat/label','label');
end

