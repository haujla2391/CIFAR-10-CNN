import data
import torch
import model

def train(batch_size, lr, epochs):

    m = model.Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)

    train_loader, test_loader = data.get_dataloaders(batch_size)

    train_losses, test_losses = [], []

    optimizer = torch.optim.Adam(params=m.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):

        m.train()
        train_loss = 0.0
        batch_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            logits = m.forward(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        avg_loss_train = train_loss / batch_count
        train_losses.append(avg_loss_train)
        
        m.eval()
        test_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                logits = m.forward(images)
                loss = criterion(logits, labels)

                test_loss += loss.item()
                batch_count += 1
        
        avg_loss_test = test_loss / batch_count
        test_losses.append(avg_loss_test)

    return m, train_losses, test_losses
