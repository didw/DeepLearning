require 'ffmpeg'

load = require 'VideoMinibatchLoader.lua'


dirname = '../UCF-101/'
dirtensor = '../UCF-101-t7/'

for dir in paths.files(dirname) do
  (function()
    if (dir == '.' or dir == '..')then return end
    path = dirname .. dir
    for f in paths.files(path) do
      (function()
        if (f == '.' or f == '..') then return end
        fname = path .. '/' .. f;
        outtensor = dirtensor .. paths.basename(fname, 'avi') .. '_';
        load.video_to_tensor(fname, dir, outtensor)
      end)()
    end
  end)()
end
