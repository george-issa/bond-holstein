I = find(A(:,1) == beta);

z = zeros(9,9); 
[x,y] = meshgrid([0:1:8]);

for ii = 1:length(I)
  idx = I(ii); 
  z(A(idx,2)+1,A(idx,3)+1) = A(idx,4);
end

z(end,:) = z(2,:);
z(:,end) = z(:,2);
z(end,end) = z(2,2);