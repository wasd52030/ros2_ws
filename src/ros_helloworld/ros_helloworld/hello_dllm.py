import rclpy

def main(args=None):
    rclpy.init(args=args)
    
    node=rclpy.create_node("hello_dllm")
    
    rate=node.create_rate(1)

    while rclpy.ok():
        node.get_logger().info("DLLM-屌你老母")

        rclpy.spin_once(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
