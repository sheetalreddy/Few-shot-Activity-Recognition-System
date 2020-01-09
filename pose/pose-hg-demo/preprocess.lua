require 'paths'
paths.dofile('util.lua')
paths.dofile('img.lua')

a = loadAnnotations('test')
idxs = torch.range(1,a.nsamples)
nsamples = idxs:nElement() 

for i = 1,nsamples do
  local center = a['center'][idxs[i]]
  local scale = a['scale'][idxs[i]]

  
  local ul = transform({1,1}, center, scale, 0, res, true)
end
