//
//  model.h
//  tflite_simple_example
//
//  Created by 夏潘安 on 11/24/30 H.
//  Copyright © 30 Heisei Google. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface DetectModel : NSObject

@property(nonatomic)float top;
@property(nonatomic)float left;
@property(nonatomic)float bottom;
@property(nonatomic)float right;
@property(nonatomic)float label;


@end

NS_ASSUME_NONNULL_END
