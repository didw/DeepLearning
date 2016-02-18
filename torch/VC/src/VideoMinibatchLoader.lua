-- loader for character-level language models

require 'torch'
require 'paths'
require 'math'
require 'ffmpeg'

VideoMinibatchLoader = {}
VideoMinibatchLoader.__index = VideoMinibatchLoader


function VideoMinibatchLoader.create(video_dir, batch_size, seq_length, class_size)
  local self = {}
  setmetatable(self, VideoMinibatchLoader)

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

  -- construct a tensor with all the data
  print('loading data lists...')

  self.current_batch = 0;
  self.evaluated_batches = 0;
  self.file_lists = {};
  self.batchsize = batch_size;
  self.seq_length = seq_length;
  self.dir = video_dir;
  self.class_size = class_size;

  for dir in paths.files(video_dir) do
    (function() 
      if dir == '.' or dir == '..' then return end
      path = video_dir .. dir
      for f in paths.files(path) do
        (function() 
          if f == '.' or f == '..' then return end
          local filepath = path.."/"..f;
          local vid = ffmpeg.Video(filepath)
          if #vid[1] - seq_length < 1 then return end
          table.insert(self.file_lists, {path.."/"..f, dir})
        end) ()
      end
    end) ()
  end
  shuffleTable( self.file_lists )

  self.nbatches = #self.file_lists / batch_size

  collectgarbage()
  print('file list load done.')
  return self
end

-- *** STATIC method ***
function VideoMinibatchLoader.video_to_tensor(in_videofile, ylabel, out_tensorfile_tmp)

    torch.setdefaulttensortype('torch.FloatTensor')
    
    local timer = torch.Timer()
    local dataset = {}
    local seq_length = self.seq_length  -- number of timesteps to unroll to

    print('timer: ', timer:time().real)
    print('loading video file...')
    local vid = ffmpeg.Video(in_videofile);
    length = math.floor(#vid[1]/8) - 1;
    print(#vid[1][1])

    -- construct a tensor with all the data
    print('timer: ', timer:time().real)
    print('putting data into tensor...', out_tensorfile_tmp)
    for i=1,length do
      dataset.data = torch.DoubleTensor(seq_length, 3, 240, 320);
      dataset.label = torch.CharTensor(1)
      for s = 1,seq_length do
        frame = (i-1)*8 + s
        dataset.data[s] = vid[1][frame];
      end
      dataset.label = ylabel;
      out_tensorfile = out_tensorfile_tmp .. i .. '.t7'
      torch.save(out_tensorfile, dataset);
    end
    print('Done in time (seconds): ', timer:time().real)
end

function VideoMinibatchLoader:next_batch()
  local seq_length = self.seq_length
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
    print('unknown label: ', label)
    return -1
  end

  local function preprocessing(dat)
    local img = dat:mul(255)
    local dim = 231
    local img_scale = image.scale(img, '^'..dim)
    local h = math.ceil((img_scale:size(2) - dim)/2)
    local w = math.ceil((img_scale:size(3) - dim)/3)
    local new_img = image.crop(img_scale, w, h, w+dim, h+dim):floor()
    return new_img
  end

  self.current_batch = (self.current_batch % self.nbatches) + 1
  self.evaluated_batches = self.evaluated_batches + 1
  -- construct a tensor with all the data
  --print('loading batch data... ', self.current_batch .. '/' .. self.nbatches)

  local x_batches = torch.FloatTensor(self.batchsize, seq_length, 3, 231, 231)
  local y_batches = torch.FloatTensor(self.batchsize);
  for i=1,self.batchsize do
    j = i + (self.current_batch-1)*self.batchsize;
    local filepath = self.file_lists[j][1];
    local vid = ffmpeg.Video(filepath)
    local len = torch.floor(torch.random(1, #vid[1] - seq_length))
    for j=1,seq_length do
      x_batches[i][j] = preprocessing(vid[1][j+len])
    end
    y_batches[i] = get_label(self.file_lists[j][2])
  end
  collectgarbage()

  --print('batch data load done.')
  return x_batches, y_batches;
end

return VideoMinibatchLoader

