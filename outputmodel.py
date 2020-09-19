import tensorflow as tf
import os
with tf.get_default_graph().as_default():
	# 定义你的输入输出以及计算图
	input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
	output_result = model(input_images, is_training=False) # 改成你实际的计算图
	
	saver = tf.train.Saver(variable_averages.variables_to_restore())

	# 导入你已经训练好的模型.ckpt文件
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
		model_path = os.path.join(FLAGS.checkpoint_path,
		os.path.basename(ckpt_state.model_checkpoint_path))
		print('Restore from {}'.format(model_path))
		saver.restore(sess, model_path)
		
		# 定义导出模型的各项参数
		# 定义导出地址
		export_path_base = FLAGS.export_model_dir
		export_path = os.path.join(
			tf.compat.as_bytes(export_path_base),
			tf.compat.as_bytes(str(FLAGS.model_version)))
		print('Exporting trained model to', export_path)
		builder = tf.saved_model.builder.SavedModelBuilder(export_path)

		# 定义Input tensor info，需要前面定义的input_images
		tensor_info_input = tf.saved_model.utils.build_tensor_info(input_images)

		# 定义Output tensor info，需要前面定义的output_result
		tensor_info_output = tf.saved_model.utils.build_tensor_info(output_result)
		
		# 创建预测签名
		prediction_signature = (
			tf.saved_model.signature_def_utils.build_signature_def(
				inputs={'images': tensor_info_input},
				outputs={'result': tensor_info_output},
				method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

		builder.add_meta_graph_and_variables(
			sess, [tf.saved_model.tag_constants.SERVING],
			signature_def_map={
				'predict_images': prediction_signature})
		
		# 导出模型
		builder.save(as_text=True)
		print('Done exporting!')