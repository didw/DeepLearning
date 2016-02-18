require 'paths'
require 'ffmpeg'

video_dir = "../UCF-101/"
file_lists = {}

for dir in paths.files(video_dir) do
  (function() 
    if dir == '.' or dir == '..' then return end
    path = video_dir .. dir
    for f in paths.files(path) do
      (function() 
        if f == '.' or f == '..' then return end
        table.insert(file_lists, {path.."/"..f, dir})
      end)()
    end
  end)()
end

local function shuffleTable( t )
    local rand = math.random 
    assert( t, "shuffleTable() expected a table, got nil" )
    local iterations = #t
    local j
    
    for i = iterations, 2, -1 do
        j = rand(i)
        t[i], t[j] = t[j], t[i]
    end
end

shuffleTable( file_lists )


local dir_lists = {
  "ApplyEyeMakeup",
  "ApplyLipstick",
  "Archery",
  "BabyCrawling",
  "BalanceBeam",
  "BandMarching",
  "BaseballPitch",
  "Basketball",
  "BasketballDunk",
  "BenchPress"
}
local function get_label(label)
  for idx, lab in pairs(dir_lists) do
    if lab == label then return idx end
  end
  return -1
end


class_size = 10

for i=1,#file_lists do
  ylabel = get_label(file_lists[i][2])
  ybatch = torch.Tensor(1, class_size):fill(0)

  for j=1,class_size do
    if ylabel == j then
      ybatch[1][j] = 1
    end
  end
  print(ylabel, ybatch)
end

