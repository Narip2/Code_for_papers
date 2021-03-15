function result = Gen_random_data(sample_num)
%result is 3 dimensional matrix
sigx = [0 1;1 0];sigy = [0 -1j;1j 0];sigz = [1 0;0 -1];
a = sigz;a_ = sigx;b = 1/sqrt(2)*(sigx-sigz);b_=1/sqrt(2)*(sigx+sigz);
result = zeros(sample_num,4);
label = zeros(sample_num,1);
for i = 1:sample_num
    rho = RandomDensityMatrix(4);
    label(i) = IsPPT(rho);
    result(i,:) = [trace(rho*kron(a,b)) trace(rho*kron(a,b_)) trace(rho*kron(a_,b)) trace(rho*kron(a_,b_))];
end
save('mat/x','result');
save('mat/label','label');
end

