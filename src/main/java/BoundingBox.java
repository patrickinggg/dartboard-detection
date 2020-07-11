public class BoundingBox {
    public int x;
    public int y;
    public int width;
    public int height;

    public BoundingBox(int x, int y, int width, int height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    @Override
    public String toString() {
        return "BoundingBox{" +
                "x=" + x +
                ", y=" + y +
                ", width=" + width +
                ", height=" + height +
                '}';
    }

    @Override
    public boolean equals(Object obj){
        if(obj == null){
            return false;
        }else {
            if(this.getClass() == obj.getClass()){
                BoundingBox box = (BoundingBox) obj;
                if (this.x == box.x) {
                    if (this.y == box.y){
                        if (this.width == box.width){
                            if(this.height == box.height){
                                return true;
                            } else {
                                return false;
                            }
                        } else {
                            return false;
                        }
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }

            }else{
                return false;
            }
        }

    }
}
