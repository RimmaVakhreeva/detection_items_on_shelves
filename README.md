### detection_items_on_shelves

**holes_on_shelves - поиск пустот на полках с товарами**

Для детекции товаров на полках используется одностадийный детектор GFL. В качестве бэкбона используется ResNet18. Модель обучалась на данных SKU110K, которые были подготовлены под формат COCO json.

Для поиска пустот были заранее выделены полки в виде полигона. Для того чтобы, соотнести каждый bounding box товара к конкретной полке была вычислена площадь пересечения bounding box c полигоном полки. Если площадь пересечения больше заданного порога, то bounding box принадлежит полке.

Все bounding boxes были отсортированы. С помощью координат bounding boxes были сохранены все пустоты между товарами. Чтобы найти нужные пустоты, были вычислены средние значения пустот по полкам и сдандартные отклонения относительно средних значений пустот. Если ширина пустоты между bounding boxes не проходит по условию, то там нет товаров.

![](result_holes_on_shelve.jpg)

**similar_products_groups - поиск группы похожих товаров**

Для поиска группы похожих товаров была обучена модель reid-strong-baseline (https://github.com/michuanhaohao/reid-strong-baseline) на датасете Aliproducts. С помощью этой обученной модели были получены embedding'и товаров. Далее к этим embedding'ам был применен метод кластеризации DBSCAN. Таким образом были найдены похожие товары.

![](result_similar_products_group.jpg)

**wrong_position_product - поиск товаров на полках, которые расположены не на своем месте**

Для решения данной задачи была обучена модель reid-strong-baseline (https://github.com/michuanhaohao/reid-strong-baseline) на датасете Aliproducts. С помощью этой обученной модели были получены embedding'и кропов товаров. Далее к этим embedding'ам к каждой выделенной полке был применен метод кластеризации DBSCAN. Таким образом были найдены товары на полках, которые расположены не на своем месте.

![](result_wrong_position_product.jpg)
