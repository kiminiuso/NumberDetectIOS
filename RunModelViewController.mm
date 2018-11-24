// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "RunModelViewController.h"

#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/op_resolver.h"

#include "ios_image_load.h"
#import "DetectModel.h"


NSString* RunInferenceOnImage();

@interface RunModelViewController ()<UINavigationControllerDelegate,UIImagePickerControllerDelegate>
@end

@implementation RunModelViewController {
}

-(void)viewDidLoad{
 
    UIImage *image = [UIImage imageNamed:@"image3.png"];
    [self.selectedImage setImage:image];
        
}

- (IBAction)getUrl:(id)sender {
//    NSString *image_name = @"image1";
//    NSString *image_ex = @"jpeg";
  NSString* inference_result = RunInferenceOnImage();
  self.urlContentTextView.text = inference_result;
}

- (IBAction)add:(id)sender {
    
    //初始化UIImagePickerController类
     UIImagePickerController * picker = [[UIImagePickerController alloc] init];
     //判断数据来源为相册
     picker.sourceType = UIImagePickerControllerSourceTypeSavedPhotosAlbum;
     //设置代理
     picker.delegate = self;
     //打开相册
     [self presentViewController:picker animated:YES completion:nil];
    
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<NSString *,id> *)info{
     //获取图片
     UIImage *image = info[UIImagePickerControllerOriginalImage];
     [self dismissViewControllerAnimated:YES completion:nil];

     self.selectedImage.image = image;
}

@end

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
  NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (file_path == NULL) {
    NSLog(@"Couldn't find '%@.%@' in bundle.", name, extension);
    exit(-1);
  }
  return file_path;
}

NSString* RunInferenceOnImage() {
//  NSString* graph = @"mobilenet_v1_1.0_224";
    NSString* graph = @"foo";
  const int num_threads = 1;
  std::string input_layer_type = "float";
//  std::vector<int> sizes = {1, 224, 224, 3};
std::vector<int> sizes = {1, 300, 300, 3};

  const NSString* graph_path = FilePathForResourceName(graph, @"tflite");

  std::unique_ptr<tflite::FlatBufferModel> model(
      tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]));
  if (!model) {
    NSLog(@"Failed to mmap model %@.", graph);
    exit(-1);
  }
  NSLog(@"Loaded model %@.", graph);
  model->error_reporter();
  NSLog(@"Resolved reporter.");

#ifdef TFLITE_CUSTOM_OPS_HEADER
  tflite::MutableOpResolver resolver;
  RegisterSelectedOps(&resolver);
#else
  tflite::ops::builtin::BuiltinOpResolver resolver;
#endif

  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    NSLog(@"Failed to construct interpreter.");
    exit(-1);
  }

  if (num_threads != -1) {
    interpreter->SetNumThreads(num_threads);
  }

  int input = interpreter->inputs()[0];
//    输入1维，输出4维
    std::vector<int> innnnnn = interpreter->inputs();

  if (input_layer_type != "string") {
    interpreter->ResizeInputTensor(input, sizes);
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    NSLog(@"Failed to allocate tensors.");
    exit(-1);
  }

  // Read the label list
  NSString* labels_path = FilePathForResourceName(@"number_labels", @"txt");
  std::vector<std::string> label_strings;
  std::ifstream t;
  t.open([labels_path UTF8String]);
  std::string line;
  while (t) {
    std::getline(t, line);
    label_strings.push_back(line);
  }
  t.close();

  // Read the Grace Hopper image.
//  NSString* image_path = FilePathForResourceName(@"grace_hopper", @"jpg");
//  NSString* image_path = FilePathForResourceName(image_name,image_extension);
     NSString* image_path = FilePathForResourceName(@"image3",@"png");
  int image_width;
  int image_height;
  int image_channels;
  std::vector<uint8_t> image_data =
      LoadImageFromFile([image_path UTF8String], &image_width, &image_height, &image_channels);
  const int wanted_width = 300;
  const int wanted_height = 300;
  const int wanted_channels = 3;
  const float input_mean = 127.5f;
  const float input_std = 127.5f;
  assert(image_channels >= wanted_channels);
  uint8_t* in = image_data.data();
//    NSLog(@"image_data%s",in);
  float* out = interpreter->typed_input_tensor<float>(0);
  for (int y = 0; y < wanted_height; ++y) {
    const int in_y = (y * image_height) / wanted_height;
    uint8_t* in_row = in + (in_y * image_width * image_channels);
    float* out_row = out + (y * wanted_width * wanted_channels);
    for (int x = 0; x < wanted_width; ++x) {
      const int in_x = (x * image_width) / wanted_width;
      uint8_t* in_pixel = in_row + (in_x * image_channels);
      float* out_pixel = out_row + (x * wanted_channels);
      for (int c = 0; c < wanted_channels; ++c) {
        out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
      }
    }
  }
  double start = [[NSDate new] timeIntervalSince1970];
  if (interpreter->Invoke() != kTfLiteOk) {
      NSLog(@"Failed to invoke!");
      exit(-1);
  }
  double end = [[NSDate new] timeIntervalSince1970];
    
  std::vector<std::pair<float, int> > top_results;
    
    const std::vector<int> ouuuuuu = interpreter->
    outputs();

    float* boxes_output = interpreter->typed_output_tensor<float>(0);
    //    等价
    //    float* outputtttt = interpreter->typed_tensor<float>(outputTTT);
    float *labels_output = interpreter->typed_output_tensor<float>(1);
    float* score_output = interpreter->typed_output_tensor<float>(2);
    float* num_detectionboxes_output = interpreter->typed_output_tensor<float>(3);
    
    const  char * s0= interpreter->GetOutputName(0);
    const  char * s1= interpreter->GetOutputName(1);
    const  char * s2= interpreter->GetOutputName(2);
    const  char * s3= interpreter->GetOutputName(3);
    
    for(int i = 0;i< 4 * num_detectionboxes_output[0];i++){
        NSLog(@"第%d个数是%f",i,boxes_output[i]);
     
    }
    
    for(int i = 0;i< num_detectionboxes_output[0];i++){
        NSLog(@"第%d个数的score是%f",i,score_output[i]);
    }
    
    NSMutableArray *list = [NSMutableArray new];
    
//    判断用的阈值
    float scoreLimit = 0.4f;
//    开始分析
    for(int i = 0 ; i < num_detectionboxes_output[0];i++){
        if(score_output[i] > scoreLimit){
            DetectModel *m = [DetectModel new];
            m.top = boxes_output[4 * i];
            m.left = boxes_output[4 * i + 1];
            m.bottom = boxes_output[4 * i + 2];
            m.right = boxes_output[4 * i + 3];
            if(labels_output[i] == 9){
                m.label = labels_output[i] - 0;
            }else{
                m.label = labels_output[i] + 1;
            }
            
            m.score = score_output[i];
            
            [list addObject:m];
        }
    }
//    排序
    if(list.count > 1){
        for (int i = 0; i < list.count - 1; i++) {
//        需要做两次判断，因为偶尔会出现某个数被识别两次，本次先不做处理，只对顺序进行排列
//        冒泡
            for (int j = 0; j < list.count -1 -i; j++) {
                DetectModel *mj = [list objectAtIndex:j];
                DetectModel *mj1 = [list objectAtIndex:j+1];
                if(mj.left > mj1.left){
                    [list exchangeObjectAtIndex:j withObjectAtIndex:j+1];
                }
            }
            
        }
    }

    
    std::stringstream ss;
    ss << "result is : ";
    for (DetectModel *m in list) {
        ss << m.label;
    }
    for (DetectModel *m in list) {
        ss << "\n";
        ss << m.label << "  score : " << m.score;
    }
    
    ss << "\nspend time :" << (int)((end - start)*1000) << "ms";
    std::string predictions = ss.str();
    NSString* result = @"";
    result = [NSString stringWithFormat:@"%@%s", result, predictions.c_str()];
  return result;
}
